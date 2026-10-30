# Stack testing

Falcon drives its ego with an external driving stack through ROS 2, runs scenario suites against it with pass/fail
criteria, and turns the results into a safety-case report against an operational design domain (ODD). The Stack
testing page and `falcon stack …` ([CLI](cli.md)) do the same things through the same code.

## CARLA 0.9.16's own ROS 2, and the official bridge

Both were checked before Falcon's bridge was built ([lessons](carla-lessons.md#native-ros-2-measured-milestone-7)
has every measurement):

- **CARLA 0.9.16's native ROS 2** (`--ros2`; Settings `server_ros2`) works on this machine: sensors a client enables
  with `enable_for_ros()` publish once per tick in synchronous mode, and `CarlaEgoVehicleControl` and
  `AckermannDriveStamped` commands drive the vehicle. But it publishes no odometry, no ego pose and no vehicle status,
  its camera_info K is wrong (fx = −36 757 for a 1280-pixel, 90° camera), its stamps are one step after its `/clock`,
  and it never waits for a stack: the client ticks on regardless, so a closed loop with a deadline per step is not
  possible with it alone.
- **The official carla-ros-bridge does not support 0.9.16**: it requires the carla package to be exactly 0.9.13 and
  exits otherwise (read from its source; its last release is 0.9.12, its last commit of July 2022). It was not run.

## Falcon's bridge

Three processes, each replaceable:

```
 stack worker (CARLA's Python)  ── TCP, JSON header + CDR payload ──  bridge node (rclpy)  ── DDS ──  your stack
 ticks CARLA, encodes the step's            one "step" per tick,           publishes the bytes        Autoware, …
 messages, waits for the control  ◀──────── control messages raw ────────  as they are; forwards
                                                                            the control topic raw
```

- The **worker** (`falcon_autonomy/workers/stack.py`) runs CARLA in synchronous mode. Per step: the scenario director
  moves its actors, the control the stack sent for the previous step drives the ego, one tick, then every sensor
  reading of exactly that frame is encoded as a ROS 2 message (plain CDR, `core/ros2msg.py`, built on the MCAP
  export's encoder) and sent to the node in one message. It then **waits for the stack's control of that step**: the
  first control message received after the step was published whose stamp is not older than the step's (a zero stamp
  counts as the current step). Older ones are counted as stale. When none comes within `deadline_s` of wall clock, the
  step is a **missed deadline** — recorded with its step number — and `on_miss` applies (`hold` the last control, or
  `brake`); the simulation never advances with a deadline still open. The first step waits `start_wait_s` (the stack
  starting).
- The **bridge node** (`falcon_autonomy/ros2/falcon_bridge_node.py`, standalone: rclpy and the message packages only)
  publishes the bytes as serialized messages (`Publisher.publish(bytes)`: nothing is decoded or copied into Python
  messages) and subscribes the control topic with `raw=True`. Run it in the `falcon-ros2-bridge:humble` image
  (`falcon stack image` builds it from the official `ros:humble-ros-base`, JetPack 6's ROS, adding the Autoware and
  Ackermann message packages from the ROS apt repository and `CarlaEgoVehicleControl.msg`) or natively on any ROS 2
  machine (`falcon stack bridge --port P --native` prints the command).
- The **link** is TCP on 127.0.0.1 (`host` and `port` in the bridge section; port 0 — the default — lets each run
  pick a free port, as a CARLA server may listen anywhere in Falcon's range). The node retries until the worker
  listens, so it may start first. `--serve` keeps one node up across the runs of a suite.

### Topics (bridge section of a stack config)

Published, all stamped with the simulation time of the frame (the World snapshot's `elapsed_seconds`; `/clock` is the
same, so `use_sim_time` stacks see one time base):

| topic | type | |
|---|---|---|
| `/clock` | `rosgraph_msgs/msg/Clock` | first in every step |
| `/tf_static` | `tf2_msgs/msg/TFMessage` | once, latched: `base_link → <sensor>_link [→ <sensor>_optical]` (the rig) |
| `/falcon/ego/<camera>/image` | `sensor_msgs/msg/Image` | `bgr8` (or `rgb8`, `bgra8`: `image_encoding`), frame `<camera>_optical` |
| `/falcon/ego/<camera>/camera_info` | `sensor_msgs/msg/CameraInfo` | K from the rig (ideal pinhole, plumb_bob, zero distortion) |
| `/falcon/ego/<lidar>/points` | `sensor_msgs/msg/PointCloud2` | x y z intensity float32, `<lidar>_link`; one full rotation, published at the step completing it |
| `/falcon/ego/<imu>/imu` | `sensor_msgs/msg/Imu` | no orientation (covariance[0] = −1) |
| `/falcon/ego/<gnss>/fix` | `sensor_msgs/msg/NavSatFix` | covariance from the rig's noise |
| `/tf` | `tf2_msgs/msg/TFMessage` | `map → base_link`, ground truth (`ego_tf: false` for a stack that localises) |
| `/falcon/ego/odometry` | `nav_msgs/msg/Odometry` | ground truth, pose in `map`, twist in `base_link` (`odometry: false` to drop) |
| `/vehicle/status/velocity_status`, `steering_status`, `gear_status`, `control_mode` | `autoware_vehicle_msgs/msg/…` | `autoware_status: false` to drop |
| `/falcon/v2x/<rsu>/cpm`, `/falcon/v2x/<rsu>/cpm_json` | `udp_msgs/msg/UdpPacket`, `std_msgs/msg/String` | each roadside unit of the world, at the steps it sends a CPM: a BTP-B header and the UPER CPM (what etsi_its_messages' conversion node decodes: `falcon v2x converter RSU`), and the CPM as JSON ([V2X](v2x.md)); `v2x: false` to drop. A node whose ROS lacks `udp_msgs` says so in its `ready` and the run leaves the topic out (noted) |

Frames are those of the MCAP export: `map` is CARLA's world mirrored (x, −y, z), REP 103 axes; steering angles are
positive to the left. The odometry and vehicle status come last in a step, so a stack that acts on the odometry has
the step's sensors first (DDS still delivers each topic on its own: a large cloud can arrive after the odometry —
the test follower waits up to 0.2 s for the cloud of the odometry's stamp). Every sensor publishes every step (the
rig's own rates are not applied); fisheye cameras and radars are not bridged. `topics` renames any topic.

Taken (`control`, one per run, on `control_topic` or the default):

| `control` | type, default topic | applied |
|---|---|---|
| `autoware` | `autoware_control_msgs/msg/Control`, `/control/command/control_cmd` | `lateral.steering_tire_angle`; `longitudinal.velocity`, `.acceleration` when `is_defined_acceleration`; stamp `Control.stamp` |
| `ackermann` | `ackermann_msgs/msg/AckermannDriveStamped`, `/falcon/ego/ackermann_cmd` | `drive.steering_angle`, `.speed`, `.acceleration` (0 = not given: the message has no flag); `header.stamp` |
| `carla` | `carla_msgs/msg/CarlaEgoVehicleControl`, `/falcon/ego/vehicle_control_cmd` | throttle, steer (CARLA's sign), brake, hand brake, reverse, as they are |

A target command becomes throttle, steer and brake by a documented law (`core/stackbridge.py`): steer = −tyre angle /
max steer angle (70° in kinematic motion; the front wheels' value in physics motion); acceleration demand
a = accel + 1.0 s⁻¹ · (speed − v) (the gain is heuristic); throttle a / 3 m/s², brake −a / 8 m/s² — exactly the
kinematic model's response, an approximation of CARLA's vehicle physics. A negative speed drives in reverse.

### What a run records

`frames.jsonl` (per step the ego's pose and speed, the control in effect during that tick at full precision, the wait
and whether the deadline was missed), `trajectories.json` and the scenario's criticality and checks, and `run.json`:
`bridge` (accepted, missed with their steps, stale, wait percentiles, the per-step costs below, the node's ROS
distribution and RMW), `performance`, the provenance. **Every run is replayable**: `falcon stack replay RUN` runs it
again applying the recorded controls step by step; in kinematic motion the replay repeats the run bit for bit
(measured on the real server: a ROS 2 run of the test stack replayed with 0 m difference for every actor).

### Measured cost

Measured on the reference machine (RTX 5080, Falcon's server capped at 8 GB, Town10HD_Opt, 20 Hz, other work running:
load average 12–20), with a 1280 × 720 camera (bgr8, 2.7 MB a step), a 32-channel lidar (600 000 points/s at 20 Hz:
a full sweep every step), an IMU and a GNSS; the test follower on Autoware's Control, the node and the follower in the
bridge image, deadline 2 s. Milliseconds per step, mean (median); `run.json` → `bridge.costs_ms` of each run:

| run | steps/s | step | CARLA tick | wait for sensors | encode | send | node publish | wait for control |
|---|---|---|---|---|---|---|---|---|
| stalled vehicle, kinematic, Fast DDS UDP only | 31.2 | 32.1 (29.7) | 12.0 (10.7) | 3.8 (1.7) | 4.6 (4.2) | 2.1 (2.0) | 1.4 (1.3) | 8.7 (5.9) |
| the same, Fast DDS default (shared memory) | 38.9 | 25.7 (21.0) | 12.7 (10.0) | 2.2 (1.2) | 4.4 (3.9) | 1.8 (1.6) | 1.3 (1.2) | 3.8 (3.4) |
| free driving, physics, UDP only, 600 steps | 44.5 | 22.5 (18.9) | 9.5 (8.8) | 0.4 (0.0) | 4.4 (4.0) | 1.3 (1.1) | 1.3 (1.2) | 6.6 (3.7) |

So the closed loop through ROS 2 costs about 13–20 ms a step on top of CARLA's tick with this rig — encoding about
4.5 ms, the link about 2 ms, the node's publishing about 1.3 ms, and the rest the round trip through DDS and the stack
(the "wait" includes the node's publishing and the follower's own work). It ran faster than real time (1.6–2.2×) and
no deadline was missed in any run. Under heavier load (another CARLA server on the GPU, load average above 20) the
same run fell to 4–21 steps/s, almost all of it in CARLA's tick and in waiting for the camera frame — CARLA's cost, not
the bridge's. The kinematic run was repeated three times on the real server with the same outcome (minimum TTC
1.39 s; the follower's smallest lidar gap 5.3 m); a 2 × 2 suite of the test stack on lead_brake and stalled_vehicle ran at 29–35 steps/s
with no missed deadline. One measurement each: timings vary from run to run on a shared machine.

## Autoware (the reference example — not run here)

Autoware could not be run on the reference machine (it needs far more memory and GPU than were free next to CARLA and
the other work). What Falcon provides for it, checked only as far as stated:

- the control message Autoware's vehicle interface consumes (`autoware_control_msgs/msg/Control` on
  `/control/command/control_cmd`) and the status it expects (`/vehicle/status/velocity_status`, `steering_status`,
  `gear_status`, `control_mode` as `autoware_vehicle_msgs`): field layouts checked byte for byte against ROS 2 Humble's
  packages (`ros-humble-autoware-control-msgs` 1.13 and `-autoware-vehicle-msgs` 1.13) by
  `tests/test_stack_ros2.py`, and the Control path driven end to end by the test follower over DDS;
- the topic mapping (`falcon stack topics PROJECT` prints it for your rig; `core.stackbridge.AUTOWARE_MAPPING`):

| Falcon | Autoware | note |
|---|---|---|
| `/falcon/ego/<lidar>/points` | `/sensing/lidar/top/pointcloud_raw` | x y z intensity only: no ring, return type or time fields |
| `/falcon/ego/<imu>/imu` | `/sensing/imu/tamagawa/imu_raw` | no orientation |
| `/falcon/ego/<gnss>/fix` | `/sensing/gnss/ublox/nav_sat_fix` | CARLA's geo-reference |
| `/falcon/ego/<camera>/image`, `camera_info` | `/sensing/camera/camera0/image_rect_color`, `camera_info` | ideal pinhole |
| `/falcon/ego/odometry` | `/localization/kinematic_state` | only when Autoware's localisation is bypassed |
| `/vehicle/status/*` | `/vehicle/status/*` | as the vehicle interface would publish |
| `/control/command/control_cmd` (taken) | `/control/command/control_cmd` | Autoware's output |
| `/clock` | `/clock` | `use_sim_time:=true` |

- `falcon_autonomy/ros2/autoware-compose.example.yaml`: the bridge node and an Autoware container on the host
  network, with the bridge section that renames the topics (`ego_tf: false`, so Autoware's localisation owns
  `map → base_link`). Autoware also needs a Lanelet2 and point-cloud map of the CARLA town, a vehicle and sensor-kit
  description matching the ego and its rig, an initial pose and a route; none of those is provided. The sensing topic
  names are Autoware's sample sensor kit's as documented upstream; check them against your release.

## The test stack

`falcon_autonomy/ros2/follower_stack.py` is the small ROS 2 stack the end-to-end tests drive with: per odometry
message it holds the line it started on (pure pursuit) and brakes for lidar points in its corridor (taken to
`base_link` through `/tf_static`), publishing one control message of any of the three types stamped with the
odometry's stamp. `falcon stack run --test-stack` or a suite's `launch` starts the bridge node and it in the image.
It is a test object, not a driver. With `--degraded-mode` it reacts to sensor failures by a minimal-risk manoeuvre
([below](#sensor-faults-mid-run-and-the-minimal-risk-manoeuvre)).

## Sensor faults mid-run and the minimal-risk manoeuvre

A stack run (or a suite entry) can carry **timed sensor faults** that start mid-drive — `faults: [{sensor, kind,
onset_s, for_s, …}]` (`core.faults.TIMED`; `falcon stack run|suite new --fault SENSOR:KIND:ONSET[:K=V,…]`):

| kind | sensor | what the stack receives from `onset_s` (for `for_s`, or to the end) |
|---|---|---|
| `occlusion` `{cover}` | camera | mud on the lens: one dark, opaque, soft-edged patch over that share of the image |
| `dirt`, `rain`, `noise`, `glare`, `compression` … | camera | the image faults of known size ([rigs](index.md#rigs)) |
| `sector_loss` `{from_deg, to_deg}` | lidar | no returns in that azimuth sector of the lidar's frame |
| `packet_drop` `{rate}` | lidar | that share of its 1° azimuth packets lost, seeded per sweep |
| `multipath` `{jump_m, period_s}` | GNSS | fixes offset by up to `jump_m` in a seeded direction, re-drawn every `period_s` (heuristic model) |
| `dropout` | any | nothing: a silent camera (no image), lidar (no cloud for a rotation with every slice silenced), IMU or GNSS |

frames.jsonl marks the steps (`faults`: the kinds acting per sensor). The faults act on the data a stack is sent — the
ROS 2 messages, the built-in follower's monitor, a HIL session's streams — not on the run's recorded files.

**The MRM verifier** (`core/mrm.py`; `mrm: true` on an entry, `--mrm`) judges the run from the first fault's onset:
did the stack bring the vehicle to a **minimal risk condition**? Every threshold is **heuristic** — a controlled stop
for Falcon's test stack, not a regulation's values:

| check | passes when |
|---|---|
| reaction | the vehicle starts slowing (≥ 0.5 m/s² for 0.3 s) within **1.5 s** of the onset |
| standstill | ≤ 0.2 m/s within 1.5 s + v_onset / 1.5 m/s², and held to the end (no restart) |
| deceleration | never above **4 m/s²** (a controlled stop, not emergency braking) |
| jerk | \|jerk\| ≤ **10 m/s³** (speed smoothed over 0.25 s), except 0.2 s in all |
| lane | the lateral offset from the scenario route stays within (lane − vehicle width) / 2 + 0.2 m, or the stop ends on the kerb side within 2.5 m of the lane's edge (the shoulder) |
| no collision | no contact of the ego after the onset |

The result is a check of the scenario (so the run passes only if the MRM does), `run.json` `mrm` with every check and
the measured reaction, standstill time, peak deceleration and jerk; a suite's `results.json` `mrm` gives the pass rate
per scenario with its Wilson interval, the failed checks and every case, `results.csv` an `mrm_passed` column, and the
safety-case report a *Minimal-risk manoeuvre after sensor faults* section (each case passed or failed, with why) and
a residual risk per scenario whose MRM failed.

**The test stack's degraded mode** (`falcon_autonomy/ros2/degraded.py`, shared by both test stacks): a health monitor
reads what arrives — a lidar silent for 0.35 s, a sweep with fewer than half the recent median points, an empty azimuth
sector wider than 45° where earlier sweeps had points; an IMU silent for 0.25 s; a GNSS silent for 0.5 s or jumping 4 m
more than the odometry moved; a camera image with over 20 % dark, texture-less 8 × 8 blocks, or 10 identical images
while moving — and once one fails (latched) the stack performs the **MRM**: the deceleration demand ramps at 4 m/s³ to
2.5 m/s², the lane-keeping steering continues, then it stands braked (thresholds heuristic). The built-in follower uses
it with `"degraded_mode": true` on the suite (`--degraded-mode`): the rig's sensors are spawned for it, their faulted
data feed the monitor, `run.json` `degraded` says when and why. The ROS 2 follower takes `--degraded-mode` (`--test-stack
--degraded-mode` adds it to the launch): it watches its point clouds and the `/falcon` IMU and GNSS topics. Without the
mode both ignore failures — the baseline the verifier must fail.

Measured against the fake carla (`tests/test_mrm.py`, 2026-10-30): with the lead never braking and the ego at 30 km/h,
a lidar dropout, a 120° lidar sector loss, an IMU dropout, a 12 m GNSS multipath jump and a 60 % camera occlusion at
3 s are each detected and the built-in follower in degraded mode **passes** every MRM check; without the mode it drives
on and **fails** (never slowed, no standstill); the ROS 2 follower's own decisions behind the bridge logic pass with
`--degraded-mode` and fail without. Over real DDS (the follower and the bridge node in `falcon-ros2-bridge:humble`
containers, the worker against the fake carla; `tests/test_stack_ros2.py`, 2026-10-30), the lidar dropping out at 3 s:
with `--degraded-mode` the follower slowed 0.55 s after the onset and stood still at 3.95 s (window 7.1 s), peak
deceleration 2.5 m/s², peak jerk 8.6 m/s³, no missed deadline — **passed**; without it, it never slowed — **failed**.
On the real server (CARLA 0.9.16, the demo project's Central Bengaluru, 2026-10-30; the built-in follower behind a
lead that never brakes, 30 km/h, a front camera 640 × 360, a 32-channel lidar at 10 Hz, IMU and GNSS, real sensor
data into the monitor), a fault at 4 s:

| fault | degraded mode | detected | reaction | standstill (window 7.1 s) | peak decel / jerk | MRM |
|---|---|---|---|---|---|---|
| lidar sector loss −60…60° | off | — | never slowed | none | — | **failed** |
| lidar sector loss −60…60° | on | 4.05 s: an empty sector of 120° | 0.25 s | 3.65 s | 2.5 m/s² / 8.6 m/s³ | **passed** |
| camera occlusion, 60 % | off | — | never slowed | none | — | **failed** |
| camera occlusion, 60 % | on | 4.00 s: 56 % of the image dark and texture-less | 0.20 s | 3.60 s | 2.5 m/s² / 8.6 m/s³ | **passed** |

Limits: the monitor's thresholds were chosen on Falcon's simulated sensors; a
frozen-image check needs motion (a standing car in a still scene sees identical images); the verifier has no lane
offsets without a scenario route (free drives, HIL sessions) and judges those on the other checks.

## Suites

A suite (`falcon.suite/1`, `core/stack.py`) is scenarios × seeds with one stack and pass/fail criteria:

```json
{"schema": "falcon.suite/1", "name": "aeb", "stack": {"driver": "ros2", "bridge": {"control": "autoware"},
 "launch": [["docker", "run", "…bridge node…", "--connect", "127.0.0.1:{port}"], ["…your stack…"]], "domain_id": 0},
 "world": "world-town10", "rig": "rig-default", "fps": 20.0, "motion": "kinematic", "traffic_vehicles": 0,
 "criteria": {"no_collision": true, "min_ttc_s": 1.0, "max_missed_deadlines": 0},
 "entries": [{"scenario": "lead_brake", "seeds": [0, 1, 2]}, {"scenario": "cut_in", "samples": 20, "seed": 7,
              "values": {"cut_gap": 4.0}, "criteria": {"min_ttc_s": 0.8}}]}
```

The stack under test is a ROS 2 stack (`ros2`), Falcon's lane follower (`follower`) or CARLA's autopilot
(`autopilot`, physics: not repeatable) as baselines. `launch` commands start with each run and stop after it
(`{port}`, `{domain}`, `{name}`, `{run}` are filled in; a named container is removed). Criteria replace a scenario's
own; a run passes when every check does. Each run is a folder named by its cache key under `<suite>/runs/`; the suite
**resumes** (finished runs are never repeated; one that did not finish — no server, no node — runs again).
`results.jsonl`, `results.json` (per scenario and overall: pass rate with its **Wilson 95 % interval**, mean
criticality with a percentile bootstrap interval over runs, minimum TTC, collisions, missed deadlines), `results.csv`
and `run.json` (the suite's hash, every run's key, provenance) are written at the end of each pass.

## The safety-case report

`falcon stack report SUITE --odd ODD.json --out DIR [--study STUDY]…` (or the page's *Build the report*) writes
`safety-case.html` (inline CSS and SVG, nothing from the network, light and dark) and `safety-case.json`:

- **The ODD** (`falcon.odd/1`, `core/safetycase.py`; `falcon stack odd new` writes an example to edit): dimensions,
  each categorical (`place`, `road_type`, `handedness`, `scenario`), a set (`actors`) or a range with bins
  (`ego_speed_kmh`, `traffic_vehicles`, `precipitation`, `fog_density`, `wetness`, `cloudiness`,
  `sun_altitude_deg`), and pairs for two-way tables. A run's place in it is read from its own config; road type is a
  heuristic (what the scenario director required at the start).
- **Coverage**: runs per value or bin, the gaps, runs outside the ODD, two-way tables.
- **Results** per scenario with their intervals, and the failing runs.
- **Triggering conditions found**: the failing runs' parameters, and for each rare-case study given the failing
  parameter sets and, per parameter, the range the failures span against the range searched.
- **Residual risks with their evidence**: each failing scenario (failure rate, interval, runs); each scenario with no
  failure whose interval still allows more than `--max-failure-rate` (5 %: a reporting threshold, not a validated
  acceptance criterion); each ODD gap; missed deadlines; the studies' critical cases; the simulation's limits.
- **Its limits**, stated in the report: it is **not a certification claim** and not a safety case in the sense of a
  complete argument; it does not cover hazard analysis and acceptance criteria, the specification's insufficiencies
  beyond the scenarios, foreseeable misuse, the unknown unsafe area outside the searched parameter ranges, real-world
  validation, hardware and software faults (ISO 26262) or cybersecurity, and its run counts are far too small for a
  statistical residual-risk argument. Simulation limits: lidar and radar ignore weather, the camera sim-to-real gap,
  kinematic motion, box-based collisions, scripted actors.

## Tests

`tests/test_stack.py` (default suite, no ROS): CDR bytes of known messages, round trips, frames and the control law,
the worker in closed loop against the fake carla with the node's own logic and the follower's decisions on an
in-process stand-in for rclpy (deadlines, stale and missed controls, on_miss), replay bit for bit, suites (resume,
intervals, table) and the report. `tests/test_stack_ros2.py` (`FALCON_ROS2=1`, Docker and the image): the encoding
against rclpy's serialiser, and the closed loop over real DDS with each control type (the worker against the fake
carla; the node and the follower in containers). `tests/test_gui_stack.py`: the page, a baseline suite as a job, a
replay and a report. `tests/test_mrm.py`: the timed fault kinds, the health monitor and the MRM profile, the verifier on
constructed runs (each failure named), the follower with and without its degraded mode on every fault kind, the ROS 2
follower's decisions through the bridge logic, a suite and its safety case, a HIL session with faults.
