# Changelog

## Unreleased

- **The assistant's runtime and panel** (`assistant/groq.py`, `provider.py`, `agent.py`, `session.py`, `service.py`,
  `grounding.py`, `keystore.py`, `ui/assistant_dock.py`; docs/assistant.md): an OpenAI-compatible streaming client on
  the standard library (Groq by default; retries with backoff and jitter honouring retry-after; every failure a
  sentence without the key; every payload redacted; nothing sent before the user turns it on and accepts the notice);
  the agent loop (read calls side by side, act and trash one at a time; Ask, Auto and Autopilot — trash always asks;
  Autopilot resumes when its jobs finish through a queue subscription; Stop closes the request in flight); a plan
  checklist and approval cards with the tool's exact preview and estimate; an activity log with Undo (project edits,
  jobs, the trash); a grounding check that marks numbers no tool returned; a conversation per project and an audit
  log of payload hashes in the workspace; Settings ▸ Assistant (the key in the keyring or the 0600 secrets file, model
  pickers from /models, the user's price table); Ctrl+Shift+A, Ctrl+K ▸ Ask Falcon…, a status-bar badge with tokens
  and estimated cost. Groq's API verified against its documentation on 2026-10-30: the fast lane is
  `qwen/qwen3.8-27b` (Preview), falling back to `openai/gpt-oss-20b` — said once — when the server does not list it or
  says it is gone; `parallel_tool_calls` per model (false for gpt-oss). Tool schemas on demand: a core of four
  meta-tools (`tools_find`, `tools_load`) and the groups loaded; each step estimated and trimmed or paced to the
  user's tokens-per-minute limit (default 8,000). Tested against a fake server only.

- **Lateral filtering for two-wheelers** (`workers/lateral.py`, a traffic profile's `lateral_filtering`, Worlds ▸
  traffic profile): background two-wheelers choose their own lateral position over the same-direction lanes every
  0.1 s — a gap-seeking chooser over 0.4 m strips with an anisotropic repulsion field and a kerb-side home offset
  (left in left-hand traffic); heuristics, recorded in run.json. Measured first: CARLA 0.9.16's
  `TrafficManager.vehicle_lane_offset` tracks ±2.4 m, but the traffic manager's collision detection ignores the offset
  both ways (stops behind a car it would clear, drives into one it overlaps). The default `tm` controller keeps the
  traffic manager for routing, lights and speed, caps the speed with its own clearance off the lane centre and switches
  the traffic manager's vehicle detection off only while filtering past an object on the lane centre. On central
  Bengaluru (6 × 60 s, 60 vehicles): close passes 2.4× (9.9 against 4.1 per 100 two-wheeler km), contacts 15.2
  against 18.1 per 100 vehicle km, falls 50 against 66, ego put-backs equal; Falcon's own pure-pursuit controller
  (`own`, kept for comparison) had 23.9 contacts per 100 km and twice the falls. On by default in the Indian profile
  for two-wheelers (not autorickshaws: their contacts rose by half). actors.jsonl `lateral`, frames.jsonl `lateral`,
  run.json `traffic.lateral_filtering`; docs/indian-assets.md.

## 2.3.0 — a living Bengaluru: city, speed limits, traffic kept, night, longer dashcam routes

- **Longer dashcam routes and rain**: a 10-minute, 8.8 km stretch of the user's daylight footage imported (from a
  local PBF), snapped (99.9 %, lateral median 0.007 m) and driven (follower: ≤ 0.36 m across, speed within 0.29 m/s
  mean); five more reconstructions placed on that world (`falcon recon place`), four a median 0.6–1.3 m from a lane
  centre. The tracker bridges wiper sweeps (fragile; heavy rain still cannot be posed) and a lost odometry says where
  and why.
- **Fix: a shared server stopped under another process's job.** A process stops its own server only while holding
  the CARLA lock, so a job of another Falcon process using it is never cut off; a server left running that way is
  stopped by the next Falcon process to shut down (`server.stop_orphan`).
- **Traffic kept in imported worlds** (`workers/trafficguard.py`, `keep_traffic`): CARLA 0.9.16's traffic manager
  destroys a vehicle whose planned path meets a dead end of its road graph (its OSM mode, on by default), and vehicles
  sink through holes in the generated road mesh — 60 vehicles in central Bengaluru became 14–17 after 150 s. The
  keeper replaces removed vehicles out of the ego's view, catches falling ones at once (also through a hole on a
  lane), and moves — or, in view, stops and later moves — vehicles heading for a dead end of the map: 58–60 alive
  throughout 180 s, none of 95 moves within the front camera's view. frames.jsonl `traffic_respawns`, run.json
  `traffic.keeper`.
- **Sidewalks labelled in imported worlds**: `Map.generate_waypoints` returns driving lanes only, so the ground
  relabelling never produced the sidewalk class; sidewalks are now found beside the driving lanes (both sides — which
  one is "right" depends on the handedness). Central Bengaluru: 4.1 % of the pixels moved from ground to sidewalk.
  The fake carla's road has sidewalk lanes.
- **The world's settings in every worker**: the learning environment, adversarial traffic and hardware-in-the-loop
  configs now carry the city, `speed_limits`, `keep_traffic` and `vehicle_lights` (`core.drive.world_keys`); vehicle
  lights are switched in Live (and again when its weather changes), stack testing, the learning environment and HIL,
  not only in drives. Measured: the city's lamp meshes and `static.prop.gardenlamp` emit no light, and 0.9.16 offers no
  way to add one — an imported world stays dark at night apart from the vehicles' own lamps.
- **A city around imported roads** (`core/cityprops.py`, Worlds ▸ City around the roads): an imported world's OSM
  buildings, trees, street lamps, bus stops and street furniture placed at load time as the package's own cooked
  meshes (`static.prop.mesh`; measured: 485 of 488 paths spawn, labelled building 3, vegetation 9, pole 6) — each
  footprint the size-nearest town building, turned and scaled to it and kept off every lane, a stand-in, not the real
  building. Read from the import's source or another file (a regional `.osm.pbf`; `core.osm.read_osm(city=True)`).
  Central Bengaluru from a regional extract: 858 buildings, 2026 trees, 329 lamps, 3341 actors spawned in 0.1–1.2 s;
  9.2 against 10.7 steps/s; building 11 % and vegetation 16 % of pixels where sky was 56 %. No plausible ground mesh
  exists in the package, so none is laid. Plans are seeded; the plan's hash is in run.json and the cache key.
- **Traffic at each road's speed limit on imported worlds** (`workers/speedcontrol.py`, `core/speedlimits.py`): CARLA's
  traffic manager ignores an OpenDRIVE file's speed records and drove everything at about 30 km/h on an imported place.
  Every autopilot vehicle (traffic, an autopilot ego, stand-ins) now gets its lane's limit from the file, less the
  profile's per-vehicle difference, and slows ahead of a lower limit or a speed breaker (a 2 m/s² profile to the
  breaker's crossing speed, heuristic). Measured on a synthetic ring: medians 19.3 / 48.3 / 63.8 km/h (p85 78.2) on
  20 / 50 / 80 km/h sectors against 29 km/h everywhere before; humps crossed at 24.0–24.6 km/h (29 before). On the
  re-imported Bengaluru, p85 79 and 90 km/h on 80 and 100 km/h lanes (38 and 35 before). A world's
  `speed_limits: "carla"` keeps the old behaviour; run.json `traffic.speed_source` says which was used. Drives of
  imported worlds get a new cache key (their config names the mode).
- **Night** (your city footage is all after dark): `falcon dashcam extract --night` (sun below −6°) and `--twilight`
  (−6° to 0°), every frame's exposure recorded; the Training page's dashcam source picks day, night, twilight or any.
  Night drives switch every vehicle's position lights and low beams on (`vehicle_lights`, run.json `lights`; CARLA
  leaves them off). The gap adds night statistics for a night real set (shadow shares, light sources, the glow round
  them, noise in the shadows). The translator's drift adds light recall and precision on night frames: removing every
  light read 0.03–0.27 and painting lamps into the dark 0.01–0.04 before — now refused on simulated frames.

## 2.2.0 — milestone 14: workbench

- **Map editor** (Worlds ▸ Edit the map…): an imported world's lanes under its OpenStreetMap ways, pan and zoom,
  select a road or junction; change lanes per direction, direction, road type and speed limit, remove a road, put
  traffic signals on a junction or a speed breaker on a road. Edits are an ordered, undoable list applied to the OSM
  data before the conversion (`core.mapedit`; a re-import keeps them), refused when a road would have no lane, with
  a warning for roads a removal cuts off; *Apply* converts again into a new folder of the world (undo returns to the
  old files) and shows the checks: the edited roads' lanes, lane positions, lane links, driving side, signals. What
  CARLA 0.9.16 honours was measured: lanes, direction, type, signals and breakers yes; a speed limit is written but
  CARLA's traffic manager does not use it (no speed-limit signs are spawned in a generated world).
- **Re-import with the current converter** (Worlds): the same area, centre, settings and edits into a new world
  beside the old one, with the lane positions, lane links and traffic-light records before and after (central
  Bengaluru: misplaced ways 11 → 1 of 23, lights controlling no lane 7 → 0). A left-hand world imported by an earlier
  version says so on the Worlds page.
- **Fix: traffic lights in left-hand worlds controlled no lane.** The left-hand conversion kept each signal's
  orientation while mirroring its lanes; CARLA then attached no lane to any light and traffic drove through red
  (central Bengaluru: 7 of 7 lights). The orientation now flips with the lanes; re-import a left-hand world to get it.
- The import reports lane links that do not meet (`lane_links`) and signal records that would control no lane
  (`signal_lanes`, `xodr.signal_lanes`) in import.json; its lane check can check given ways (`xodr.lane_check(ways=…)`).

- **Search everything from Ctrl+K**: the command palette also finds runs, scenarios, worlds, rigs, models, drive sets,
  evaluations, reports, settings, the manual's headings and past jobs — grouped by kind, the best group first — and
  opens each on its page (a run in Data, a scenario selected, a setting focused, the manual at the heading).
  Workspace listings are cached until their folder changes.
- **Job history and notifications**: every finished job is kept in the workspace (`jobs/jobs.jsonl`, with its whole
  log in `jobs/logs/`) and listed in the Jobs panel's History tab across restarts, with Open and Show the log. A job
  of 20 s or more that ends while the window is in the background raises a desktop notification (Settings ▸ Job
  notifications; never offscreen).
- **Scenario timeline editor** (Scenarios ▸ Timeline): events on a time axis — drag a time trigger or a lane change's
  duration (snapped), zoom, playhead — a form for the trigger and actions, add, duplicate and remove, every change
  undoable and validated inline (a trigger missing its fields is refused; unknown actors and actions are flagged), and
  a plan of everyone's paths on a straight schematic road (the director's rules, no physics; labelled a plan, not a
  simulation). `core.scenarios.validate` now also checks trigger kinds and their fields and action kinds.
- **Run search and drive sets** (Data ▸ Browse, `falcon runs search`): every run of the workspace — drives, scenario and
  search runs, replays, dashcam routes — indexed from what it recorded (world, weather and its kind, time of day,
  scenario and values, duration, speed, distance, collisions, teleports, ODD exits, saved rule checks), cached by file
  times; sortable columns and a query language (`world:w-blr weather:rain status:ok dur>20 after:2026-10-28
  tag:golden -kind:search`). Tags on runs live in the project (undoable; runs untouched). Drive sets — fixed lists or
  saved queries re-evaluated on use — are picked above every list of drives (Models, Training, VLM, world model,
  deployment). Record can set a weather preset for one drive.
- **Compare** (Data ▸ Compare): two to four runs side by side on one clock, aligned by simulation time (different
  rates and lengths line up), their events on a shared timeline, and what each recorded with the difference from the
  first — rule violations and lane margins from a background rule check; "not recorded" where a run has no number.
- **Dataset health** (Data ▸ Health): class balance, frames per world × time of day × weather with the empty cells,
  scenario values over their ranges, near-duplicate frames (dHash, 4 bits) and box sanity, measured on evenly sampled
  frames; *Record this…* opens Record with the missing cell's world and weather.
- **Drive your dashcam route** (`falcon dashcam route`, Worlds ▸ Drive a dashcam route…): the streets a clip's GPS
  track drives are imported (from a local OSM or PBF file; a download from Overpass only after confirmation), the track
  is snapped to the imported lanes, driven in CARLA as an ordinary recorded run (its run.json names the clips and the
  window) and paired frame by frame with the footage by GPS time. On one 60 s clip of the user's: 99.5 % snapped,
  lateral median 0.39 m; the follower tracked the recorded speed within 1.05 m/s mean absolute.
- **Home** is the start page: the next step the project needs (a world, a rig, a dataset, a model, an evaluation,
  then the rare-case search), the project's counts, the server and the machine, the recent runs, evaluations and
  jobs — every item a link. Read from the project and the workspace only.
- **Navigation**: the navigator groups its pages (Build, Drive, Improve, Ship); Ctrl+0 Home, Ctrl+1 … Ctrl+9 the next
  nine pages; the command palette is at the navigator's foot.
- **Layout**: side panels widen to what their forms need, so no field is cut off; a tall side pane scrolls, so the
  window fits a 1280 × 720 screen; combo boxes no longer size to their longest item; spin boxes read left-aligned in
  the body font. The Jobs and Log panel stays hidden until a job starts (the status bar's Jobs badge or Ctrl+J shows
  it), with no second title.
- **Theme**: a type scale (page 18 px, section 14 px, body 12 px); check boxes in lists visible in the dark theme; the
  page's main action is the primary button; a disabled control says why in its tooltip.
- **Text**: a page or section opens with one line, the rest of its explanation behind *more* — nothing removed.
- **Empty states and hand-offs**: empty lists and result pickers say what goes there; Data ▸ Browse opens on the
  newest run, filters runs, and hands selected drives to *Evaluate on these* and *Train on these*; Worlds has *Drive
  this world* and *Record a dataset here*; the import dialog *Add and drive it*; Models ▸ Failures *Open the run in
  Data* (at the failing frame); a finished job's *Open* goes to its result. An imported place hides the Town field;
  the road map's caption no longer covers the roads; the dashcam folder shows from ~.
- **Left-hand imports: one-way roads in the right place.** The conversion mirrored every road about its reference
  line, but the OSM converter draws a one-way road's reference line at its carriageway's edge, so one-way roads moved
  a whole carriageway sideways (central Bengaluru: 36 of 245 roads, 3.35–13.4 m; a dual-carriageway highway: GPS on
  a lane 0.7 % instead of 59.9 %). Each road is now mirrored about the line its OSM way runs along, and junction
  roads get a laneOffset that meets their links. The lane check also measures lane positions against OSM. Worlds
  imported before need importing again.
- **Poses land in their own step.** `set_transform` is asynchronous in CARLA's client: a log replay showed the
  previous step's ego pose in 22 of 1181 frames on a real server. Replays, kinematic scenarios, the gym's start and
  the road-end guard now place actors in one synchronous batch before the tick (0 of 1181).

## 2.1.0 — your own dashcam

- **Dashcam ingest** (`falcon dashcam list|gps|extract|match`, [dashcam](docs/dashcam.md)): reads the GPS that
  consumer dashcams embed in their MP4s (`freeGPS` blocks: position, UTC, speed over ground in knots, course; the
  layout checked against the fixes' own motion and the burned-in overlay), decodes frames on NVDEC, extracts seeded
  training and held-out sets filtered by speed, sun altitude and GPS, and places each clip against an imported world
  (inside the box, distance, map match). Footage and frames stay in the workspace; nothing private is in the repo.
- **Translation to your own roads**: `dashcam` is a real-set kind for the gap and for CUT translation (Training ▸
  Sim-to-real translation ▸ Dashcam). Measured on the user's daytime footage (FD floor 31.2): Town10 151.2 → 148.0
  and the imported Bengaluru 218.9 → 190.6 at 6000 iterations; the default training repaints vehicles (Town10) or
  invents fields (Bengaluru) and the drift check refuses it.
- **Reconstruction from a dashcam** (`falcon recon from-dashcam`, Replay ▸ From a dashcam): monocular visual odometry
  (KLT, PnP, Falcon's own Levenberg–Marquardt bundle adjustment) held to scale by the GPS speed, aligned to the GPS,
  focal length measured from a turn, then 3D Gaussians. On KITTI raw 0015 (363 m) 1.42 m RMS against OXTS; on three
  20 s segments of the user's footage 3.5–11.3 m RMS odometry-only against the GPS, held-out 23–25 dB below the
  horizon (26–28 dB whole frame, inflated by sky). 7–11 min per segment on an RTX 5080.
- **Road-end guard**: an autopilot ego that leaves every driving lane at the edge of an imported area and falls is
  put back on a road (a `teleport` in frames.jsonl) instead of crashing the live view.

## 2.0.0 — milestone 13: deployment

- **TensorRT on the Jetson AGX Orin** (`falcon deploy`, Models ▸ Deploy): a device-side tool copied over SSH builds
  FP32, FP16 and INT8 engines with trtexec (INT8 with TensorRT's entropy calibrator on Falcon frames, cached by a
  SHA-256 key; DLA with GPU fallback, fallback layers reported), profiles latency and throughput, samples power from
  the INA3221 rails (energy per frame = mean power × median latency, labelled estimated), runs the engine on
  evaluation frames and scores it against the host's ONNX model with a paired drive-level bootstrap; a release gate
  (`falcon.deploy-gate/1`: relative AP by the interval's lower bound, p95 latency, energy, power mode) gives pass,
  fail or incomplete with evidence. Built against TensorRT 10's documented API and trtexec's flags and exercised end
  to end only on a **simulated device** (onnxruntime on this host, labelled everywhere): it has not run on an Orin —
  that needs the device's address and key. `models.nms` now sorts stably (numpy 1.21 ordered ties differently).
- **CI mode** (`falcon ci`): a `falcon.ci/1` config tests a ROS 2 stack, a policy or a model against gates — pass rate
  by Wilson's interval, metric bounds, no regressions by paired bootstrap against a stored baseline, rule violations
  per km, latency, missed deadlines — with JUnit XML, a Markdown summary (also to GitHub's job summary), an HTML report
  and the commit under test; exit 0 pass, 1 gate failed, 2 error, 3 cancelled. `falcon ci init --github` writes a
  workflow (a fake-carla smoke job on GitHub's runners, a real job on a self-hosted GPU runner); validated against
  GitHub's workflow schema, not yet run on GitHub. Posting is a dry run without `--post`.
- **Vision-language scorer 2**: truth limited to what the camera sees, absence claims scored as such, the ego never a
  hallucination; Qwen2-VL-2B (downloaded on approval) scores 0.299 [0.243, 0.350] on two real Town10 drives — no
  better than a constant answer on any question type.

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
