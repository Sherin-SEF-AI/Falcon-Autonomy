# Falcon Autonomy manual

Falcon Autonomy is one application for the loop of building a self-driving AI on CARLA 0.9.16:
**generate → train → test → find failures → generate more exactly there.** This manual covers what the current build
does; the [roadmap](roadmap.md) lists what is planned.

## Getting started

1. **First run.** The wizard finds CARLA (a folder holding `CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping`) and
   its Python (a venv beside it with the carla 0.9.16 wheel), checks the GPU, the NVIDIA Vulkan driver, memory and
   `systemd-run`, then runs a **30 s smoke drive**. What the smoke drive measures — steps per second, megapixels
   rendered per second, the real-time factor — becomes this machine's throughput for every cost estimate after.
   Tools ▸ First-run wizard opens it again.
2. **A project.** File ▸ New starts one; File ▸ Save writes `name.falcon.json`. Its runs, datasets and models go in
   the workspace folder beside it (`name.falcon/`). An unsaved project's runs go to the cache folder until it is
   saved.
3. **Record a drive.** Data ▸ Record: choose the world and the rig, the duration, the step and the seed. The cost is
   shown before it starts: the megapixels the rig renders per step against the render budget, and the time it will
   take, labelled *measured* (this machine) or *typical* (the reference machine, before the smoke drive).
4. **Browse it.** Data ▸ Browse: every recorded run, frame by frame, with the semantic layer over the image at any
   opacity, a legend of the classes in the frame with their share of pixels, and the run's measured performance and
   provenance.

## The window

- **Navigator** (left): Home, then the pages in groups — **Build** (Worlds, Rigs, Scenarios), **Drive** (Live, Data,
  Replay), **Improve** (Loop, Rare cases, Training, Models), **Ship** (Stack testing, Hardware, Reports) — and Settings
  and Help. Ctrl+0 goes Home, Ctrl+1 … Ctrl+9 to the next nine pages; the rest by the command palette.
- **Jobs and Log** (bottom): every long action — queued, waiting (and why), running with its progress and ETA,
  finished — and what every job and the app said. The panel stays out of the way until a job starts, and opens
  then; the **Jobs** badge in the status bar (or Ctrl+J) shows or hides it, and says how many jobs run and wait. The ✕
  cancels a job (a cancelled worker tidies the server first); a finished job's **Open** (or a double-click) goes to its
  result — a recorded drive in Data ▸ Browse, anything else on the page it was started from. **History** lists every
  finished job of the workspace, kept across restarts (`WORKSPACE/jobs/jobs.jsonl`: kind, title, start and end in
  UTC, status and what its exit code means, its result, its last error line), with **Open** and **Show the log**
  (each job's whole log is kept in `WORKSPACE/jobs/logs/`; the newest 2000 jobs are kept).
- **Notifications**: a job of 20 s or more that ends while Falcon's window is in the background raises a desktop
  notification (through the desktop's notification service, else the system tray) — off in Settings ▸ Job
  notifications. Nothing leaves the machine.
- **Status bar**: the jobs badge, Falcon's CARLA server (start or stop it there), the GPU's memory and load, free
  memory.
- **Command palette** (Ctrl+K, also at the foot of the navigator): every command and page by name — and the
  project's and workspace's things: runs, scenarios, worlds, rigs, models, drive sets, evaluations, reports, settings,
  the manual's headings and the job history. Type a few letters (in order; word starts and runs of letters rank
  higher, a title starting with them highest); results come grouped by kind, the best group first. Enter opens the
  highlighted one on its page — a run in Data ▸ Browse, a scenario selected in the studio, a setting with the
  keyboard on it, a manual page scrolled to the heading; ↑ ↓ and Page Up/Down move through every group.
- **Explanations**: a page or section opens with one line; *more* reads the rest in place (the method, the numbers,
  the limits — nothing is left out). A greyed-out button says in its tooltip why it is off. An empty list says what
  goes there and how to get some.
- **Undo and redo** (Ctrl+Z, Ctrl+Shift+Z): every change to the project. Dragging a slider is one step.
- **Themes**: View ▸ Dark theme / Light theme.

## Pages

### Home
The start page: the **next step** this project needs, in the order the work goes (a world, a rig, a recorded dataset,
a model, an evaluation, then the search for what it gets wrong), with a button to where it is done; the project's
worlds, rigs, scenarios and models (each a link); Falcon's CARLA server (start, stop, probe) and the machine's GPU and
memory; the recent runs (double-click: Data ▸ Browse), evaluations (double-click: Models) and this session's jobs
(double-click a finished one: its result). Everything on it is read from the project and the workspace; it computes
nothing.

### Loop
The closed loop ([loop](loop.md)). Choose a **model** and a target **operational design domain** — the
worlds, weathers (with ranges, such as dusk's sun altitude), traffic profile, classes of interest and range bins; *Use
the Bengaluru monsoon-dusk example* writes one and says what the project still needs — then run the eight steps one at
a time or up to a step, each a job with progress, ETA and cancel: **pick** (the model and ODD are frozen and hashed),
**evaluate** (on a held-out evaluation set recorded once per ODD at fixed seeds and never trained on), **weak cells**
(AP per class × range bin × condition with intervals; a cell is weak when its whole interval lies below the overall AP
— a heuristic), **search** (a rare-case search on a scenario matched to the cell's class, in its weather; the objective
is the scenario's criticality, not the model's misses), **generate** (a dataset spec targeting the weak cells, planned
from the yield measured on the evaluation set, and the failing cases re-run with labels), **retrain** (`falcon train`
from the model's own settings on its data plus the new drives; a model Falcon did not train cannot be, and the step
says so), **evaluate again** and **before and after** (the same evaluation set; paired intervals over drives). The
cost of the next step is shown before it runs — from this machine's measured throughput and trainings, or labelled
typical, or *not estimated* with the reason — and recorded beside the measured time. The weak cells, the before-and-
after chart with intervals and the history of iterations are below; each iteration is a record in
`workspace/loop/NNN/iteration.json`, and a finished step is never run again unless what it was made from changed.

The **Status** tab: the server (state, port, folder, client Python, the last probe), the machine's checks, the
throughput Falcon uses for estimates and where it came from, and the latest runs (double-click one to browse it).

### Worlds
A world is a CARLA town with its weather and time of day, and the traffic profile it drives with. *Import a place…*
and *Add town* sit above the list of worlds. Presets set all of CARLA's weather fields; sliders override them one by
one. *Reads as* says what the weather amounts to (day or night, rain, fog visibility). An imported place has no town
to choose, so its Town field is not shown. *Drive this world* opens Live with it chosen; *Record a dataset here* opens
Data ▸ Record with it chosen; the import dialog's *Add and drive it* adds the place and goes to Live. An imported
place has *Edit the map…* — change a road's lanes, direction, type or speed limit, remove it, put traffic signals on
a junction or a speed breaker on a road; every edit undoable, applied by converting the area again, with the lane
checks after — and *Re-import with the current converter…*, which converts the same area into a new world beside the
old one and compares their checks ([worlds](worlds.md): what CARLA does with each edit).

**Traffic profiles** are named and shared between worlds: vehicle and walker counts, fleet shares per CARLA base type,
behaviour (gaps, speed spread, ignoring lights and signs, random lane changes, walkers crossing anywhere) and an
asserted handedness. *Add the Indian mixed-traffic profile* adds a left-hand profile with a high share of
two-wheelers, buses, trucks and small cars; see [worlds](worlds.md) for its limits (the autorickshaw substitute).
A run whose profile's handedness differs from the map's lanes is refused before it starts.

Limits, stated where weather is chosen: **CARLA's lidar and radar ignore weather** (no loss of range in rain or
fog); rain on a camera is water on the lens and wet surfaces, not loss of range; **fog visibility ≈ 7000 m / fog
density** (measured on 0.9.16, a 5 % contrast threshold — a human observer's; for a detector it is an upper bound).

The town list is the base package's (Town01–05, Town10HD and their `_Opt` variants) until Tools ▸ Probe the CARLA
server lists the server's own maps.

Each world can switch on **sensor weather physics** — what CARLA's sensors leave out: lidar and radar losses and
clutter in rain and fog, lens droplets, glare and bloom — modelled from cited sources, labelled, recorded in every run
([sensor physics](sensor-physics.md)). **Road features** write speed breakers into an imported world's road surface
(from OpenStreetMap or every N metres), and Indian traffic gets measured stand-ins and proxies within what the packaged
CARLA can spawn ([Indian road assets](indian-assets.md)). **City around the roads** places CARLA's own buildings,
trees, lamps and street furniture where OpenStreetMap has them — size-matched stand-ins kept off the lanes, labelled
correctly ([worlds](worlds.md#a-city-around-the-roads)).

### Live
Drive a world in real time. **Start** opens a session (a job like any other: the Jobs dock shows it and Stop ends it).

- **View**: chase (behind the ego), free (I/K/J/L move, U/O up and down, drag to look) or any pinhole camera of the
  rig; the view's size is chosen before starting (1280 × 720 by default). **Ground-truth boxes**: every vehicle's and
  walker's full 3D box projected into the view (occlusion not considered). The **top-down map** shows the lanes and
  the actors around the ego, north up; the wheel zooms.
- **Driving**: *Manual* by the keyboard (click the view first: W/S or ↑/↓ throttle and brake, A/D or ←/→ steer, Space
  hand brake, Q reverse) or a game controller or steering wheel (any device under `/dev/input/js*`; choose which axis
  steers, accelerates and brakes, and whether the pedals rest at −1). If no control reaches the session for 0.5 s the
  ego brakes — a lost input never drives on. *Autopilot* is CARLA's traffic manager. *Policy* is a plug-in policy
  ([plug-ins](plugins.md)); the built-in lane follower (pure pursuit) is a baseline that ignores traffic lights and
  cross traffic.
- **Live events**: a pedestrian steps out ahead, a car cuts in from the next lane in the same direction (when there is
  one), the weather changes, a sensor fails for 3 s (black, frozen or dropped frames).
- **Recording**: the rig's cameras with their labels, the controls and the events, as a run of its own that Data ▸
  Browse opens with every label layer.
- **Measured**: simulation steps per second and the real-time factor (from the worker), frames shown per second here,
  and dropped frames — by the worker when the window had not taken the last frame yet, and here when a newer frame
  arrived before the last was drawn. A step is never skipped: when the simulation cannot keep up, the real-time factor
  falls below 1. Measured on the reference machine: one 1280 × 720 chase view at 20.0 frames/s, real-time factor 1.0,
  no drops; 19.7 frames/s while also recording a 1280 × 720 camera with semantic labels.

### Rigs
A rig is CARLA's car with its sensors: pinhole and fisheye cameras, lidar, radar, GNSS, IMU. Set each pose by numbers
or drag the sensor in the 3D view (Shift: height); the view orbits on a drag and zooms with the wheel (OpenGL, or a
software fallback). Each sensor shows its **cost** — the megapixels it renders per step against the render budget —
and the rig is checked for problems (duplicate ids, poses far from the car, sizes, lidar fields of view). A fisheye
is rendered as the pinhole views its lens needs (cube faces, 3° of overlap feathered across the seams) and remapped.

**Labels** per camera: any of semantic, instance, depth, 2D boxes, 3D boxes renders CARLA's instance and depth cameras
beside each view; lidars record point labels from a semantic lidar. **Import / export**: Falcon's rig JSON, ROS URDF
and static TF, nuScenes calibration (CARLA's left-handed frame converted through rotation matrices).

**Faults of known size** per sensor: a mount turned or moved, a clock offset, dropped or frozen frames, noise, dirt or
rain on the lens, glare, rolling shutter, compression, motion blur. Outputs keep the planned calibration, as a real
faulty rig's would; each frame says which faults touched it.

### Scenarios
The scenario studio: the library (eleven built-ins, four of them for Indian roads) and the project's own scenarios; a
sketch of the scene with the actors where the seed's sample puts them and each parameter's range as a band; the
actors, the parameters as distributions and the timeline. **Run this sample** runs one; **Sweep** runs many,
unattended, and resumes where it stopped. OpenSCENARIO import and export. The **Timeline** tab is an editor: the
events on a time axis (drag a time trigger or a lane change's duration, snapped; the wheel zooms), the selected
event's trigger and actions in a form, and above it a **plan** of where everyone goes — a plan on a straight
schematic road, not a simulation. The **Trigger rules** tab holds the project's live rules — `when ego.speed > 40 and
visibility < 30 -> spawn jaywalker` — evaluated every tick of drives, scenario runs and Live, every firing recorded.
See [scenarios](scenarios.md).

### Rare cases
Search a scenario's parameters for where it becomes critical (Bayesian optimisation, CMA-ES, adaptive sampling or
random): the curve of the search, the failing cases ranked — replay one, export it as OpenSCENARIO, browse its run —
and the failure region over two parameters.

Rare-case search can also aim at the **model's** failures (*Objective: perception*): each evaluation runs a
registered model on the scenario's frames and scores how it misses the actors that matter before the critical moment
([model-in-the-loop search](rare-cases-mil.md)). **Why did this fail?** re-runs a failing case changing one factor at a
time — occlusion, range, light, fog, rain, appearance, sensor faults — and says which were necessary or sufficient, in
that case ([root cause](root-cause.md)).

### Models
Register a perception model (ONNX, TorchScript or a plug-in; its licence and pinned SHA-256 shown, AGPL refused,
weights downloaded only when asked), **evaluate** it on checked drives — AP/AR or mIoU with a 95 % bootstrap interval
over whole drives, breakdowns by range, occlusion, truncation and height, AP over a grid of two conditions, the
worst frames with their misses and false positives (*Open the run in Data* shows a failing frame in its drive) — and
**compare** two evaluations (a paired bootstrap: a regression is a drop whose whole interval is below zero). **Sim-to-real gap**: the drives' images against a real
dataset (KITTI, nuScenes, BDD100K, COCO or a folder), statistic by statistic and in a fixed feature space, each
distance next to its real-vs-real floor; optional appearance matching with its effect measured. See
[models](models.md) and [sim-to-real](sim-to-real.md). World-model and language data are derived from drives by
`falcon derive` ([world model](world-model.md)).

### Training
- **Detectors** fine-tunes a torchvision detector (BSD-3-Clause) on checked drives and optional real data. A held-out
  drive picks the best epoch, the model is exported to ONNX with a check against torch, and *Register as a model*
  puts it on the Models page. Pretrained weights download only when asked, with their terms shown.
- **Sim + real experiment** trains real only against real + X % synthetic at the same budget. Each arm is scored on the
  same real test set, with paired intervals against real only and the spread over seeds; the experiment exports for
  MLflow ([training](training.md)).
- **Driving policies** clones the recorded control of drives, or learns by PPO in CARLA along a scenario curriculum.
- **Leaderboard** scores a policy as the CARLA Leaderboard scores routes (driving score, route completion, infraction
  penalty, with intervals). Select any infraction and *Replay* it: the episode is re-run exactly with a chase camera
  ([policies](policies.md)).

### Data
**Record** a drive (see *Getting started*); *Weather* can set a CARLA preset for that drive alone, the world keeping
its own. **Browse** recorded runs — drives, scenario and search runs, replays and dashcam routes — with their date,
world, weather, duration, status and tags in sortable columns, and a **search** that reads like
`world:w-blr weather:rain status:ok dur>20 after:2026-10-28 tag:golden` (hover the box for every term). Choose the
camera and the layer, scrub the frames, set the overlay's opacity; it opens on the newest run. Select one or more runs
(Ctrl/Shift-click, or right-click) and *Evaluate on these* (Models ▸ Evaluate with them checked), *Train on these*
(Training ▸ Detectors), *Compare*, *Tags…* (kept in the project, the runs untouched) or *Save as a set…*; *Save
search…* keeps the search itself as a **drive set** that is re-evaluated when used. Every list of drives (Models,
Training, the VLM, world-model and deployment tabs) has a *Drive set* picker that checks a set's drives and saves the
checked ones as a new set. A cancelled or failed run keeps only `run.json` (why it ended) and its input, never half a
dataset.

**Compare** puts two to four runs side by side on one clock — aligned by simulation time from each run's first frame,
so runs at different rates and lengths line up — with their events (collisions, rule violations, teleports, ODD exits,
scenario triggers) on a shared timeline and a table of what each recorded, with the difference from the first; what a
run did not record says so. **Health** measures a set of drives: class balance, frames per world × time of day ×
weather (the empty cells), scenario values over their ranges, near-duplicate frames and box sanity, and suggests what
to record next — *Record this…* opens Record with that world and weather. See [runs](runs.md).

**Datasets**: a dataset spec sets targets — objects per class, over worlds × weathers — and the engine **plans** the
drives that fill them (how many, from the yield measured on the dataset's own drives, or a stated prior before any),
shows their **cost**, and **runs** them; each drive's folder is named by its cache key, so re-running resumes and
never repeats a finished drive. The **coverage** dashboard shows what the dataset has against its targets, per class
and distance bin. A **version** freezes the finished drives with a train/validation/test split by drive or by place
that never puts one drive (or world) in two splits. **Browse** shows any label layer (semantic, instance, range,
boxes) and the frame's objects. The format is in [dataset format](dataset-format.md).

The run browser checks a drive against **traffic rules** written in temporal logic (the Indian set: keep left,
overtake on the right, stop at red, yield at crossings, a two-second gap, junctions clear; or a right-hand set, or your
own): a timeline of violations with how close each rule came to breaking ([traffic rules](traffic-rules.md)). The
Training page's **Active learning** tab measures which frames help a detector most, against random selection
([active learning](active-learning.md)).

**Runtime ODD monitoring**: put the operational design domain in the workspace (`odd.json`, calibrated with
`falcon odd calibrate`) and every drive, live session, episode, stack run and HIL session flags when its conditions
or its images leave it ([ODD monitoring](odd-monitoring.md)).

**Real to sim**: the Training page's *Sim-to-real translation* tab trains a geometry-preserving translator (CUT) on
drives against real images and writes translated runs whose labels stay valid, with a drift check per frame
([translation](translation.md)); the Replay page reconstructs a recorded drive as 3D Gaussians and re-renders it from
new viewpoints ([reconstruction](reconstruction.md)). Your own dashcam footage — its GPS, frames (by day, at night
or at twilight), translation targets and reconstructions — is read in place ([dashcam](dashcam.md)).

**Learned worlds**: the Training page's *World model* tab trains a latent dynamics model on drives and episodes and
scores it per horizon against simple baselines, learns policies in its imagination and clones camera-to-control
policies ([world-model training](world-model-training.md)); the Rare cases page trains **adversarial traffic** that
tries to make the ego fail under realism constraints ([adversarial traffic](adversarial-traffic.md)); the Models
page's *Language* tab scores vision-language models' driving answers against the simulator's truth
([vision-language](vlm.md)).

### Stack testing
Connect an external driving stack through ROS 2: Falcon's bridge publishes the rig's sensors and waits each step for
the stack's control (Autoware Control, Ackermann or CARLA's own message), recording missed deadlines. Run scenario
suites with pass/fail criteria and intervals, and write a safety-case report against an operational design domain —
in the spirit of ISO 21448, not a certification claim. What CARLA 0.9.16's native ROS 2 does and does not do, as
measured, is in [stack testing](stack-testing.md).

### Replay
Import a real or recorded log (nuScenes, Waymo v2, ROS 2 bags, KITTI raw, Falcon runs), see where it fits a world and
how well its path matches the lanes, replay it in CARLA, change it (timing, speeds, remove an actor, add a pedestrian,
weather) and compare the original with the changed run ([log replay](log-replay.md)).

### Hardware
Stream a camera (H.264 over RTP, or raw) and a lidar to a device — the Jetson AGX Orin, or a simulated device on this
machine — take its control back, and measure latency stage by stage with a waterfall; the device's address and key
live only in your settings ([hardware](hardware.md)).

### Deployment
The Models page's *Deploy* tab builds TensorRT engines for a model on the Jetson AGX Orin, profiles latency, power and
energy, checks accuracy against the host's ONNX model with paired intervals and applies a release gate
([deploy](deploy.md)). **CI mode** (`falcon ci`) runs a stack's, policy's or model's suite headless on every commit with
gates, JUnit and Markdown outputs, and a GitHub Actions template; posting to GitHub needs an explicit flag
([CI](ci.md)).

### Reports
Self-contained HTML reports ([reports](reports.md)) of a loop iteration, a model evaluation, a dataset's coverage, a
rare-case study or a Leaderboard result: inline CSS and SVG (nothing fetched from the network), light and dark, every
number with its interval where it has one, and the provenance — the source's own and the SHA-256 of every file the
report read. Choose the kind and the source, *Make the report* (a job), then open it in the browser or export a
selection to a folder.

### Settings
Where CARLA and its Python are; Falcon's ports, the server's memory cap and quality; the render budget; the theme;
whether place-name search may use the network. Stored in `$XDG_CONFIG_HOME/FalconAutonomy/settings.json`, never in a
project.

**Ports.** Falcon starts its own server on its own RPC port (default 2310; CARLA also listens on the next two) and
traffic-manager port (8310). It never touches a server it did not start, nor ports 2000–2010 and 8100 (CARLA's
defaults, where your own server usually listens) — unless you choose such a port here. Then Falcon connects to the
server already running there and never starts or stops it.

### Help
This manual. The search box filters the pages and highlights the first match.

## The CARLA server

Falcon starts CARLA headless (`-RenderOffScreen`) in its own systemd user scope capped at the memory you set (12 GB by
default) with no swap, so a runaway server cannot take the machine down, with `VK_ICD_FILENAMES` pinned to NVIDIA's
Vulkan driver (else Unreal can pick the integrated GPU that drives the desktop). It is started when the first CARLA
job needs it and stopped when Falcon quits.

**One CARLA job at a time** per GPU: a machine-wide lock queues CARLA jobs — from the GUI and the CLI alike — and a
GPU job (training, evaluation) never runs while a CARLA job holds the GPU.

**Memory.** Before Falcon starts its server it checks that the cap plus a 2 GB reserve is free (one server plus a
heavy job needs about 22 GB); otherwise the job waits, saying why, and starts when memory frees up.

**After a crash.** A worker whose server stops answering is ended by its watchdog (exit code 4) with `run.json`
saying why. A worker killed outright cannot tidy up; Falcon then tidies the server (Tools ▸ Tidy the CARLA server
does it by hand): it removes exactly what Falcon workers spawned and restores asynchronous mode.

## Render budget and cost

A camera costs the pixels it renders: its image once, plus once per label camera (semantic, instance, depth). A
fisheye renders five cube faces. A 0.9.16 server capped at 12 GB crashed at 306 Mpx per step on the reference machine
and ran at 77: runs over the **render budget** (200 Mpx by default) are refused before they start. See
[methods](methods.md) for the cost model.

## Reproducibility

Every run folder holds its input (`drive.json`), what was recorded, and `run.json`: the versions, the seed, the
settings, the inputs' hashes, the measured performance, the status and exit code. `falcon reproduce RUN` re-runs it
from its own input and compares the result within a tolerance (see [methods](methods.md#reproduction)).
