# Falcon Autonomy

One desktop application for the whole loop of building a self-driving AI on the
[CARLA](https://carla.org) 0.9.16 simulator: build or import a world, drive it, script and search scenarios,
generate perfectly labelled data where models are weak, train and evaluate, and test a full AV stack — with every
number traceable to the run that produced it.

**Status: 2.0 — the eight milestones of the plan and the five that followed (9–13).** What works is listed under *What works*; what comes next is in
the [roadmap](docs/roadmap.md). The app shows only pages that do what they say.

## What works (2.1.0)

- **The app shell** (PyQt6): a page navigator, Jobs and Log docks, undo and redo for every project edit, a command
  palette (Ctrl+K), dark and light themes, a status bar with the CARLA server, the GPU and memory.
- **The project file**: JSON with a [published schema](falcon_autonomy/schemas/project.schema.json), a format
  version and migrations ([format reference](docs/project-format.md)). Large data lives in a workspace folder beside
  it, indexed by a local SQLite catalogue.
- **Falcon's own CARLA server**: headless, in a systemd scope capped in memory with no swap, on its own ports, with
  the NVIDIA Vulkan driver pinned. Falcon never touches a server you started, nor ports 2000–2010 and 8100, unless
  you choose them in Settings.
- **Background jobs** with progress, ETA and cancel; an app-wide (machine-wide) CARLA lock so one CARLA job runs at a
  time; a memory guard that queues rather than overcommits.
- **Workers** in CARLA's own Python, speaking a [documented JSON-lines protocol](docs/protocol.md), each writing a
  `run.json` and exiting with a [documented code](docs/cli.md#exit-codes). A watchdog ends a worker whose server
  stopped answering; cancels always tidy the server.
- **Recorded drives**: synchronous mode at a fixed step, the ego and traffic on the traffic manager, the rig's
  cameras with semantic labels rendered at the same pose and tick; browse them frame by frame with the label overlay.
- **Rigs and data**: a rig editor with a 3D view; pinhole and fisheye cameras, lidar, radar, GNSS, IMU; sensor faults
  of known size; perfectly labelled frames (semantic, instance, range, 2D and 3D boxes, occlusion, track ids, lane
  lines, lidar point labels proven point for point); a data engine that plans, costs and fills coverage targets;
  exports to COCO, YOLO, KITTI, nuScenes, Waymo-style, BDD lanes and ROS 2 MCAP, checked with the official devkits
  ([dataset format](docs/dataset-format.md)).
- **Scenarios and rare cases**: a scenario studio with eleven built-ins (four for Indian roads), parameters as
  distributions, runs that repeat bit for bit with the same seed (measured), criticality measures (TTC, PET, minimum
  distance, required deceleration), resumable sweeps, rare-case search (Bayesian optimisation, CMA-ES, adaptive
  sampling) with ranked, replayable failures and a failure-region map, OpenSCENARIO import/export
  ([scenarios](docs/scenarios.md)).
- **Models and the gap**: register a detector or segmenter (ONNX, TorchScript or a plug-in; licence shown, weights
  downloaded only on request and checked by SHA-256), score it on drives with COCO AP/AR or mIoU and bootstrap
  intervals over whole drives (pycocotools' numbers exactly), breakdowns by range, occlusion and size, a condition
  heat map, a failure gallery and paired regression tests between versions ([models](docs/models.md)); the
  sim-to-real gap against KITTI, nuScenes, BDD100K or COCO images — appearance statistics, Fréchet and kernel
  distances next to their real-vs-real floor, and measured appearance matching ([sim-to-real](docs/sim-to-real.md));
  optical flow, scene flow, occupancy with visibility, BEV maps, future trajectories, clips and grounded captions and
  QA derived from drives ([world model](docs/world-model.md)).
- **The training lab**: fine-tune torchvision detectors on drives and real data (pinned weights on request, seeded,
  resumable, exported to ONNX and checked), the mixed sim/real experiment with paired intervals, local tracking with
  an MLflow export ([training](docs/training.md)); a CARLA environment with the Gymnasium API, behaviour cloning, PPO
  with a scenario curriculum, Leaderboard-style driving scores and one-click replay of any infraction
  ([policies](docs/policies.md)).
- **The Loop** (the home page): pick a model and an operational design domain, evaluate, find the weak cells, search
  for rare cases there, generate targeted data, retrain, evaluate again on the same held-out set and show before and
  after with intervals — each iteration a versioned, resumable record with its costs ([loop](docs/loop.md)); HTML
  reports with provenance ([reports](docs/reports.md)).
- **Sharper loop** (1.1): rare-case search aimed at the model's own misses; *Why did this fail?* by counterfactual
  re-runs; active learning against random selection; traffic rules in temporal logic (an Indian set) checked on every
  drive, episode and stack run ([model-in-the-loop search](docs/rare-cases-mil.md), [root cause](docs/root-cause.md),
  [active learning](docs/active-learning.md), [traffic rules](docs/traffic-rules.md)).
- **Realism** (1.2): sensor weather physics CARLA lacks (lidar and radar in rain and fog, droplets, glare), runtime
  ODD monitoring with a calibrated false-alarm rate, and Indian road assets within the packaged CARLA — autorickshaw
  and cycle-rickshaw stand-ins by measured size with cited kinematics, speed breakers in the road surface, cattle and
  handcart proxies, waterlogging ([sensor physics](docs/sensor-physics.md), [ODD monitoring](docs/odd-monitoring.md),
  [Indian road assets](docs/indian-assets.md)).
- **Real to sim** (1.3): a geometry-preserving sim-to-real translator (CUT, trained here) whose translated runs keep
  their labels, checked per frame; neural reconstruction of recorded drives as 3D Gaussians (pure PyTorch) re-rendered
  from new viewpoints, including from your own dashcam's footage and GPS ([translation](docs/translation.md),
  [reconstruction](docs/reconstruction.md), [dashcam](docs/dashcam.md)).
- **Learned worlds** (1.4): a latent world model scored honestly against baselines, policies learnt in imagination,
  end-to-end camera policies; adversarial traffic that learns to make the ego fail within realism limits;
  vision-language driving answers scored against the simulator's truth ([world-model training](docs/world-model-training.md),
  [adversarial traffic](docs/adversarial-traffic.md), [vision-language](docs/vlm.md)).
- **Deployment** (2.0): TensorRT engines built, profiled (latency, power, energy) and accuracy-checked on the Jetson
  AGX Orin with a release gate; CI mode with gates, JUnit/Markdown/HTML outputs and a GitHub Actions template
  ([deploy](docs/deploy.md), [CI](docs/ci.md)).
- **Stack testing**: a closed-loop ROS 2 bridge (Autoware, Ackermann or CARLA control), scenario suites with pass/fail
  and intervals, a SOTIF-style safety-case report; CARLA 0.9.16's native ROS 2 measured and documented
  ([stack testing](docs/stack-testing.md)).
- **Log replay**: nuScenes, Waymo v2, ROS 2 bags, KITTI raw and Falcon runs rebuilt in CARLA, changed (timing, speeds,
  an added pedestrian, weather) and compared ([log replay](docs/log-replay.md)).
- **Hardware in the loop**: camera (H.264/RTP) and lidar streamed to a Jetson (or a simulated device), control back,
  latency per stage with a waterfall ([hardware](docs/hardware.md)).
- **A Linux bundle** (PyInstaller, torch and CUDA included) ([packaging](docs/packaging.md)).
- **Worlds**: CARLA's towns with weather presets and sliders (sun, cloud, rain, wetness, puddles, fog, wind), and
  **real places imported from OpenStreetMap** (.osm or .osm.pbf, offline) with a road-graph preview, a lane check
  against OSM, **left-hand traffic** (verified on the server) and the **Indian mixed-traffic profile**
  ([worlds](docs/worlds.md)).
- **Live**: drive in real time by keyboard, controller or wheel, autopilot or a policy plug-in; chase, free and sensor
  views with ground-truth boxes and a top-down map; live events; recording as labelled runs. 1280 × 720 at 20 frames/s
  measured.
- **The first-run wizard**: finds CARLA, checks the GPU, the Vulkan driver and memory, runs a 30 s smoke drive and
  records this machine's measured throughput for every cost estimate after.
- **A CLI** for every heavy action, on the same code paths as the GUI, including `falcon reproduce`.

## Requirements

- Linux x86_64 (developed on Ubuntu 26.04), an NVIDIA GPU with its Vulkan driver, systemd (user scopes).
- CARLA 0.9.16 (not bundled; Falcon finds it) and a Python venv beside it with `carla==0.9.16`, `numpy` and
  `opencv-python-headless` — the workers run there.
- For the app: Python ≥ 3.12 with PyQt6, numpy, OpenCV, jsonschema, pyqtgraph and PyOpenGL.

## Running

```sh
python3 -m falcon_autonomy                 # the application (the first run opens the wizard)
python3 -m falcon_autonomy doctor          # check the machine and CARLA
python3 -m falcon_autonomy project new Demo ~/demo.falcon.json
python3 -m falcon_autonomy smoke ~/demo.falcon.json
python3 -m falcon_autonomy drive ~/demo.falcon.json --seconds 30 --seed 7
python3 -m falcon_autonomy reproduce ~/demo.falcon/runs/<run>
```

## Tests

The whole suite runs without a GPU or a server, against a fake `carla` package (`tests/fake_carla`) whose instance
ids match its boxes, whose depth matches its lidar and whose semantic lidar matches its lidar. Run it memory-capped
with a throw-away config folder (the suite sets `XDG_CONFIG_HOME` itself):

```sh
systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 python3 -m pytest
FALCON_REAL_CARLA=1 python3 -m pytest -m real_server     # also against a real CARLA server (starts Falcon's own)
```

## Documentation

[The manual](docs/index.md) · [architecture](docs/architecture.md) · [protocol](docs/protocol.md) ·
[project format](docs/project-format.md) · [CLI and exit codes](docs/cli.md) · [methods](docs/methods.md) ·
[CARLA lessons](docs/carla-lessons.md) · [models](docs/models.md) · [sim-to-real](docs/sim-to-real.md) ·
[world model](docs/world-model.md) · [training](docs/training.md) · [policies](docs/policies.md) · [loop](docs/loop.md) · [reports](docs/reports.md) ·
[stack testing](docs/stack-testing.md) · [log replay](docs/log-replay.md) · [hardware](docs/hardware.md) ·
[packaging](docs/packaging.md) · [deploy](docs/deploy.md) · [CI](docs/ci.md) · [roadmap](docs/roadmap.md) · [changelog](CHANGELOG.md)

## Licence

GPL-3.0-or-later. Falcon ports its theme, icons, fonts (IBM Plex, SIL OFL) and parts of its CARLA bridge and fake
`carla` from Vantage (also GPL-3.0-or-later, same author).
