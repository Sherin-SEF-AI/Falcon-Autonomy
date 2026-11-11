# Hardware in the loop

Hardware ▸ Session (or `falcon hil`) streams one camera and one lidar of a CARLA session to a device in real time — a
Jetson AGX Orin on the network, or a **simulated device** on this machine — runs a stack there, takes its control back
and applies it to the ego at the next simulation step. Every frame's latency is recorded stage by stage, from the
step that rendered it to the step that applied the control computed from it, and shown as a **latency waterfall**.

The code: `core/hil.py` (configs, emulated sensor clocks, the clock-offset estimate, the statistics),
`workers/hil.py` (the session worker), `hil/gateway.py` (the streams), `hil/device_agent.py` (the device side, one
file), `hil/report.py` (the report), `hil/deploy.py` (SSH), `hil/cli.py` (`falcon hil`), `ui/pages/hardware.py`.
Since 4.0 also: `hil/v4l2out.py` (the camera into a V4L2 device), `hil/gmsl.py` (V4L2 capture on the device, the GMSL2
injector interface), `core/ptp.py` (PTP and the clock source), `core/chassis.py` (the chassis model),
`core/canmsgs.py` (the CAN message table and `resources/can/falcon_chassis.dbc`), `hil/canbridge.py` (the chassis on
a CAN bus), `ui/settings_hardware.py` (Settings ▸ Hardware) — sections [CAN](#can), [the chassis
model](#the-chassis-model), [the camera as a V4L2 device](#the-camera-as-a-v4l2-device), [GMSL2](#gmsl2),
[clocks: PTP and the NTP-style fallback](#clocks-ptp-and-the-ntp-style-fallback) and [setting up the
interfaces](#setting-up-the-interfaces) below.

## How it runs

Three processes, each in the Python that has what it needs:

| process | Python | does |
|---|---|---|
| session worker (`workers/hil.py`) | CARLA's venv (carla, numpy) | steps the world in synchronous mode at a fixed rate paced to the wall clock; reads the camera and the lidar of exactly that frame; stamps them; hands them to the gateway over a local socket pair; applies the newest control that came back before the next step |
| gateway (`hil/gateway.py`) | the app's (GStreamer through PyGObject) | encodes the camera, frames the lidar, sends both to the device, measures the clock offset to it, passes the control back |
| device agent (`hil/device_agent.py`) | the device's (3.10+, numpy, python3-gi) | receives and decodes, pairs the camera and the lidar of each step, runs the stack, sends the control back with its timestamps |

CARLA's venv has no GStreamer bindings (checked: `import gi` fails there), and Falcon installs nothing into it; so the
worker hands raw frames to the gateway (a Unix socket pair, the live link's framing) — a hop that is measured
(*hand-off*: 1.3–1.8 ms median at 1280 × 720 with a 32-channel lidar here, see below).

The simulation never waits for the device, as the world would not: a control that comes back after the next step
started is applied at the one after (*late*). With no control yet the ego is held braked; with none for
`control_timeout_s` (0.5 s) it brakes — a lost device never drives on. The device's stack sees the ego's speed in each
lidar message (an emulated CAN speed).

## Streams and formats

Ports: the agent listens on TCP and UDP port **P** (default 7710; the session, the clock pings); the camera's RTP goes
to UDP **P + 1**, a UDP lidar to **P + 2**. Every framed message is `>II` (header length, payload length,
big-endian) + a UTF-8 JSON header + the payload — the live link's framing.

**Camera** (`--camera`):

- `h264-rtp` (default): H.264 over RTP/UDP (RFC 6184, payload type 96, MTU 1400, SPS/PPS with every IDR, an IDR every
  second). Encoder `nvh264enc` (NVENC: preset p1, tune ultra-low-latency, zero latency, no B-frames, CBR) or `x264enc`
  (tune zerolatency, speed-preset ultrafast, no B-frames, sliced threads); `auto` takes NVENC when present; 8 Mbit/s by
  default. Each access unit carries a **user-data-unregistered SEI** (UUID `FalconAutonomy/1`) holding
  `{"f": frame, "s": sim_t, "h": hw_t}` as JSON, inserted before the first slice: a decoded frame is matched to its
  step exactly, and any H.264 receiver (gst-launch, VLC, a ROS 2 camera driver) plays the stream and ignores it. The
  device receives with `udpsrc ! rtph264depay` (**no jitter buffer**: it would add its latency; a lost packet loses its
  frame) and decodes with `nvv4l2decoder ! nvvidconv` on a Jetson, else `avdec_h264` (slice threads, no frame delay),
  `nvh264dec` or `openh264dec`, to BGRx.
- `raw-tcp`: raw BGR (8 bits, row-major) on a TCP stream, one framed message per step:
  `{"t": "camera", "frame", "sim_t", "hw_t", "w", "h", "format": "bgr8"}` + the pixels. 1280 × 720 at 20 Hz is
  442 Mbit/s: for comparison, and for a device without GStreamer.

**Lidar** (`--lidar`): one message per step — the slice of the rotation CARLA delivers that step —
`{"t": "lidar", "frame", "sim_t", "hw_t", "n", "encoding", "speed_mps"}` + the points:

- `xyzi` (default): float32 x y z intensity per point, the sensor's frame (CARLA's: x forward, y right, z up);
- `pointcloud2`: a `sensor_msgs/PointCloud2` in CDR, encoded by `core/mcap_bag.py` (the same encoder as the MCAP
  export) — what `--ros2-topic` uses.

Over `tcp` (default) on its own TCP stream, or over `udp` cut into datagrams of at most 1400 bytes, each with a
16-byte header (`FLDR`, message id = the frame, chunk index, chunk count, total length); a message is used only when
every chunk arrived, and an incomplete one is dropped when a newer one completes. A 32-channel slice is about 0.5 MB:
over UDP that is ~350 datagrams in a burst, so raise the device's receive buffer (`sudo sysctl -w
net.core.rmem_max=8388608`; the agent asks for 8 MB).

**ROS 2** (`--ros2-topic /falcon/lidar`): the agent republishes every lidar message unchanged as a PointCloud2 on that
topic (rclpy's publish of serialised bytes, sensor-data QoS, frame `lidar`, stamped with the emulated hardware clock).
It needs ROS 2's environment sourced before the agent starts. Verified with the official `ros:humble-ros-base`
container (opt-in test): all 30 lidar messages of a session published, and a subscriber in the container received them
with their fields and frame intact. The camera does not go over ROS 2 here
(image transport is the stack-testing bridge's job), and the control comes back over the session, not a topic.

**Control** (device → host, on the session connection): `{"t": "control", "frame", "throttle", "steer", "brake",
"missing": [...], "info": {...}, "dev": {cam_rx, dec_start, dec_end, lidar_rx, stack_start, stack_end,
ctrl_sent}}`.

## Timestamps and clocks

All stamps are int64 nanoseconds. On the host, CLOCK_MONOTONIC (the worker and the gateway share it); on the device,
the device's CLOCK_MONOTONIC — an unrelated clock.

**Emulated sensor clocks.** Each camera and lidar message also carries the simulation time of its step and a
"hardware" timestamp from an emulated free-running sensor clock:

    hw_t = round(1e9 · (offset_s + (1 + drift_ppm · 1e-6) · sim_t) + N(0, jitter)) ns        (seeded)

set per sensor (`--camera-clock` / `--lidar-clock OFFSET_MS,DRIFT_PPM,JITTER_US`, or *Sensor clocks* on the page), so
a stack can be tested against realistic clocks. The report fits a line through each sensor's stamps against the
simulation time and shows the recovered offset, drift and jitter next to the configured ones (measured below:
offset within 0.015 ms and drift within 0.02 ppm over 60 s).

**The clock offset to the device** (NTP's exchange). The gateway pings the agent over UDP ten times a second (and 30
times before the first frame): it sends at t0 (host), the agent stamps receipt t1 and reply t2 (device), the gateway
receives at t3 (host).

    offset θ = ((t1 − t0) + (t2 − t3)) / 2     (device = host + θ)     round trip δ = (t3 − t0) − (t2 − t1)

If both directions take equally long, θ is exact; whatever the asymmetry, the truth lies within θ ± δ/2. Queueing only
adds delay, so in each window (2 s, or a sixth of a shorter session) the ping with the least δ is kept, and a line
θ(t) = a + b·(t − t_ref) is fitted through those (b is the drift between the clocks). The **stated uncertainty** is half
the median least round trip plus the fit's residual RMS: a bound if no delay is negative, plus the fit's scatter — not
a confidence interval. **Its limit**: a constant asymmetry (one direction always slower) is invisible to any such
exchange and shifts θ by half the asymmetry; the bound still holds, the estimate is off within it (shown by the tests:
a 0.3 ms asymmetry shifts the estimate by 0.15 ms).

So the report separates:

- stages on **one clock** — everything on the host, everything on the device — exact (to the clocks' resolution);
- the **round trip** (camera sent → control received, host only) and the **time on the device** (received → control
  sent, device only): exact, and their difference is the **network round trip**, also exact;
- the **one-way** network stages (to the device, back): need θ. They are marked *trusted* only when the uncertainty
  is below 1 ms; otherwise the waterfall hatches them and the table says so. Over a LAN with sub-millisecond round
  trips they are trusted; over a congested Wi-Fi they would not be.

A simulated device can be given an emulated clock (`--sim-offset-ms`, `--sim-drift-ppm`, *Simulated clock* on the
page): its clock is this machine's, offset and drifted, so the truth is known and the report checks the estimate
against it.

## Latency stages

On the camera's path (the lidar is not encoded, so the camera is the critical path; the lidar's own send and network
time are reported beside it):

| stage | from → to | clock |
|---|---|---|
| Simulation step | the step starts (`world.tick()`) → it returns | host |
| Sensor data to the client | the step returned → the camera and the lidar of that frame are in hand | host |
| Hand-off to the gateway | → the gateway has the frame (worker → gateway socket) | host |
| Encode and packetise | → the access unit (SEI inserted) handed to the RTP payloader / the raw frame onto the socket | host |
| Network to the device | → the device has the whole access unit / frame | one-way |
| Decode on the device | → the decoded BGRx frame (for raw: the array view) | device |
| Waiting for the other sensor | → the stack starts (both sensors of the step are there, or the sync time-out of one step passed) | device |
| Stack | the stack's `act()` | device |
| Control leaves the device | → handed to the socket | device |
| Network back to the host | → the gateway has the control | one-way |
| Control to the simulation | → the worker has it | host |
| Waiting for the next step | → applied, at the start of the next step | host |

**End to end (sensor to actuation)** is their sum: from the start of the step that rendered the frame to the start of
the step that applied its control. In a real-time session it is quantised to whole steps — at 20 Hz a control that is
back in time is applied 50 ms after its step began — so the report also shows **step start to control at the
simulation**, the continuous latency. A control is **on time** when applied at the very next step (the deadline is
that step's start); **late** when at a later one; **superseded** when a newer one arrived first; **missing** when
none came back; **after_end** when it came back after the session's last step (not counted).

Each stage: median, p95, p99, mean, standard deviation (its **jitter**), max. Also: the step interval's jitter, the
camera's RFC 3550 interarrival jitter at the device (send and receive times each differenced on its own clock),
frames lost per stream, frames dropped by the gateway (each sender keeps only the newest frame waiting), the device's
counters, steps that ran late (pacing restarts from now: never a burst to catch up) and steps braked for lack of
control.

## The waterfall

One row per stage in the order a frame goes through them; each bar starts where the previous stage's median ended and
is as long as its own median, so the rows add up to the median frame; a whisker reaches the stage's p95. The last row
is the end-to-end latency with ticks at its p95 and p99 (the sum of stage medians is not the median of the sum: both
are shown). Dashed lines mark whole steps. Bars are coloured by where the stage runs — this machine, the network,
the device — and every row is labelled. On the Hardware page it updates once a second (the last two seconds' frames)
and shows the whole session at the end; `report.html` (self-contained) and `waterfall.png` are written into the run
folder, and `falcon hil report RUN` rewrites them from the run's files — also for a session that was cancelled.

## The simulated device

*Simulated device* (`--device local`, the default) runs the same `device_agent.py` as a separate process on this
machine, on 127.0.0.1 and free ports, started and stopped by the gateway, its log in the run folder
(`device-agent.log`). Everything is exercised end to end — encoding, RTP, the SEI, decoding, the stack, the control,
the clock exchange — except a real network and the Jetson's own decoder and CPU: the network stages are loopback, and
decoding is `avdec_h264` on this machine's CPU. The agent also runs under Python 3.10 and numpy 1.21 (Ubuntu 22.04's,
as on JetPack 6) in the official `ros:humble-ros-base` container: a session closes the loop there (opt-in test).

## The default stack

`brake_for_obstacles`: drives at a target speed (6 m/s) and brakes for anything in its lidar corridor ahead (±1.4 m,
0.3–2.5 m above the ground, 2.6–40 m ahead of the lidar, the last full rotation kept), at a stopping gap of 6 m; with no
lidar for 0.5 s it brakes. It steers straight. It is a **trivial controller** whose only purpose is to close the loop
honestly — its control depends on what the device received — not a driving policy. It also reads the decoded camera
(its mean brightness, in the control's `info`) to prove the image arrived. Replace it on the device with
`--stack mymodule:MyStack`: a class with `__init__(self, params, session)` and `act(self, camera, lidar, meta)`
returning `throttle`, `steer`, `brake` (camera: H × W × 3 or 4 uint8 BGR(x) or None; lidar: N × 4 float32 or None;
meta: frame, sim_t, speed_mps, the hardware stamps). The host never sends code: the stack is chosen on the device.

## Sensor faults in a session

`hil_config(…, faults=[{sensor: 'camera' | 'lidar' | its id, kind, onset_s, for_s, …}], mrm=True)` injects timed faults
into what is streamed, from `onset_s` seconds after the first step: the camera's image faults (`occlusion` — mud —,
`dirt`, `rain`, `noise` …) on its pixels, a dropped-out camera as black frames (the stream keeps its rate), the lidar's
`sector_loss`, `packet_drop` and `dropout` on its points (a dead lidar sends empty sweeps). `latency.jsonl` marks the
frames (`faults`); with `mrm` the session is judged by the minimal-risk-manoeuvre verifier over the recorded speeds
(`run.json` `mrm`; [stack testing](stack-testing.md#sensor-faults-mid-run-and-the-minimal-risk-manoeuvre); no lane
check: a free drive has no route). The default stack above has no degraded mode — an empty sweep counts as a sweep —
so it **fails** that check (measured with the simulated device, `tests/test_mrm.py`): the device's stack must bring its
own reaction.

## Deploying to the AGX Orin (JetPack 6)

The device's address, the agent port, the SSH user, the key's path, the SSH port, the folder and the device's Python
are kept in your settings (`$XDG_CONFIG_HOME/FalconAutonomy/settings.json`, `hil_device`), never in a project or a
run folder: a remote session gets the address in its worker's environment, and run.json says only "from the user's
settings". No password is stored or asked for — Falcon logs in with your key only (BatchMode; a new host key is
accepted on first contact, a changed one refused).

On the Jetson (once):

```sh
# usually present on JetPack 6:
sudo apt install python3-numpy python3-gi gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0
sudo nvpmodel -m 0 && sudo jetson_clocks          # optional: maximum performance, stable clocks for measuring
```

On this machine:

```sh
ssh-keygen -t ed25519            # if you have no key
ssh-copy-id -i ~/.ssh/id_ed25519.pub USER@ORIN
falcon hil device --address ORIN --ssh-user USER --ssh-key ~/.ssh/id_ed25519
falcon hil deploy                 # copies device_agent.py to ~/falcon-hil and prints what the device has
falcon hil acceptance my.falcon.json    # the acceptance run: one command
```

Or on the Hardware page: fill in the device, *Save*, *Copy the agent and check*, then *Run the acceptance*. Without
SSH, copy the one file yourself and run `python3 device_agent.py` there (`--once` for one session); then
`falcon hil session PROJECT --device remote`.

**The acceptance** (`falcon hil acceptance`, *Run the acceptance*): copy and check; start the agent for one session;
60 s on the project's first world with its rig's first camera at 1280 × 720 and 20 Hz as H.264 (NVENC here, decoded by
`nvv4l2decoder` there) and a 32-channel lidar over TCP; the control back; the report; the agent stopped. The device
needs inbound TCP and UDP on ports P to P + 2 (Ubuntu on JetPack has no firewall by default).

`falcon hil deploy --start [--keep-running]` / `--stop` start and stop the agent by hand (stopped by its pid only).

## Measured here, and what is pending on the Jetson

Everything below was **measured on this host with a local simulated device**: an AMD Ryzen 9 9900X (24 threads),
an RTX 5080 (driver 595.99), GStreamer 1.28.2, Falcon's own CARLA 0.9.16 server (8 GB cap, Epic quality), Town10HD,
10 traffic vehicles, seed 7, 60 s at **20 Hz** (1200 steps), the camera at **1280 × 720** (90°) and a **32-channel
lidar** (100 m, 10 rotations/s; 600 000 points/s, about 29 000 points and 0.47 MB per step), H.264 at 8 Mbit/s, the
lidar over TCP, the agent decoding with `avdec_h264` on this machine's CPU. **The machine was shared**: its 1-minute
load average was 9–15 during these sessions (other projects' test suites). The network stages are loopback.
2026-10-28; runs `nvenc`, `x264`, `raw`, `nvenc-light`, `x264-light`.

Median / p95 in ms:

| stage | NVENC | x264 | raw BGR (TCP) | NVENC, lidar 100k pts/s | x264, lidar 100k pts/s |
|---|---|---|---|---|---|
| Simulation step + sensor data to the client ¹ | 14.6 / — | 13.4 / — | 13.2 / — | 11.7 / — | 11.7 / — |
| Hand-off to the gateway | 1.78 / 4.46 | 1.70 / 3.27 | 1.69 / 3.79 | 1.32 / 3.21 | 1.33 / 4.30 |
| Encode and packetise | 3.06 / 6.47 | 4.57 / 7.27 | 5.08 / 9.66 ² | 2.40 / 4.25 | 4.03 / 12.20 |
| *of which the encoder alone* | 1.66 / 4.06 | 3.35 / 5.49 | — | 1.39 / 2.40 | 3.07 / 10.10 |
| Network to the device (loopback, one-way) | 0.37 / 1.60 | 0.31 / 0.60 | 0.80 / 1.39 | 0.30 / 0.91 | 0.28 / 1.38 |
| Decode on the device (`avdec_h264` + to BGRx) | 3.81 / 9.16 | 1.29 / 2.63 | 0.02 / 0.03 | 3.43 / 6.22 | 1.18 / 4.97 |
| Waiting for the other sensor | 0.85 / 2.30 | 0.77 / 1.50 | 0.04 / 0.08 | 0.70 / 1.31 | 0.69 / 1.78 |
| Stack (brake_for_obstacles) | 0.49 / 0.73 | 0.43 / 0.64 | 0.43 / 0.63 | 0.19 / 0.29 | 0.20 / 0.35 |
| Network back to the host (one-way) | 0.14 / 0.35 | 0.14 / 0.25 | 0.11 / 0.21 | 0.12 / 0.24 | 0.12 / 0.36 |
| Control to the simulation | 0.09 / 0.26 | 0.09 / 0.15 | 0.06 / 0.12 | 0.08 / 0.15 | 0.07 / 0.21 |
| **Step start to control at the simulation** | **27.97 / 73.50** | **23.67 / 47.13** | **22.52 / 44.90** | **20.77 / 41.85** | **20.17 / 60.10** |
| Round trip, camera sent → control back (exact) | 5.85 / 13.05 | 3.04 / 5.46 | 1.44 / 2.31 | 4.81 / 9.23 | 2.52 / 9.11 |
| Time on the device (exact) | 5.25 / 11.31 | 2.56 / 4.64 | 0.50 / 0.76 | 4.37 / 7.79 | 2.11 / 6.77 |
| Network round trip (exact, loopback) | 0.52 / 2.06 | 0.46 / 0.85 | 0.92 / 1.56 | 0.43 / 1.18 | 0.40 / 1.91 |
| Lidar framed and sent / its network | 1.15 / 0.24 | 1.07 / 0.21 | 5.28 / 0.25 ² | 0.81 / 0.10 | 0.81 / 0.10 |
| **End to end (sensor to actuation)** median / p95 | 49.94 / 110.40 | 49.94 / 50.01 | 49.94 / 50.01 | 49.94 / 50.11 | 49.95 / 99.69 |
| On time (of 1199) | 992 (82.7 %) | 1143 (95.3 %) | 1147 (95.7 %) | 1161 (96.8 %) | 1078 (89.9 %) |
| late / superseded / no control | 135 / 72 / 0 | 14 / 42 / 0 | 6 / 46 / 0 | 5 / 33 / 0 | 77 / 44 / 0 |
| Real-time factor; steps that ran late | 0.985; 14 | 1.000; 0 | 1.000; 0 | 1.000; 0 | 0.999; 1 |
| Step interval jitter (std) | 6.44 | 1.81 | 1.45 | 1.20 | 2.81 |
| Camera interarrival jitter (RFC 3550) | 0.22 | 0.13 | 0.17 | 0.30 | 0.09 |

¹ CARLA returns from the step before the camera's data reaches the client; which of the two stages carries the render
depends on the load (with the lighter lidar the step itself took 5.7 ms and the data 6.0 ms; with the full lidar 13–14
and 0.3–0.6 ms). Only their sum is meaningful. ² Raw: converting BGRA to BGR and framing 2.7 MB per step, and the
lidar's sender waiting for the gateway's interpreter meanwhile.

What these say:

- **The loop closes within a step**: the median control is back at the simulation 20–28 ms after its step began, well
  inside the 50 ms period, so the median end to end is exactly one step (49.94 ms). No frame was lost on either
  stream in any session; no control went missing.
- **Deadline misses come from the simulation's tail, not the link**: the late and superseded controls follow steps that
  took long (the step's p95 is 30–52 ms on this shared machine); the device and the network stay at a few
  milliseconds even at p95. The NVENC session with the full lidar ran during a load peak (the step's p95 51.6 ms, 14
  steps late, real-time factor 0.985).
- **NVENC encodes about twice as fast as x264** at 1280 × 720 (1.7 vs 3.4 ms), but its single-slice stream decodes
  about three times slower in `avdec_h264` on the CPU (3.8 vs 1.3 ms; x264's sliced stream decodes on several
  threads). On the Jetson `nvv4l2decoder` decodes in hardware, so this comparison will not carry over: measure it there.
- **Raw BGR** removes encoding and decoding but costs 5 ms to frame 2.7 MB a step and 442 Mbit/s on the link — on a
  gigabit LAN about 22 ms of wire time per frame, which loopback does not show.
- **The clock-offset estimate** (simulated device clock offset 1500 ms and drifting 20 ppm, so the truth is known):
  within 5.2–7.0 µs of the truth in every session, with stated bounds of ±24–30 µs; the drift came out 19.97–20.05 ppm.
  The least round trip on loopback was 16–34 µs. One-way stages were trusted throughout.
- **The emulated sensor clocks** (camera: 2 ms, 30 ppm, 50 µs jitter; lidar: −1 ms, −15 ppm, 10 µs): recovered from
  their stamps to within 0.015 ms in offset and 0.02 ppm in drift; the jitter came out 47.9 and 9.9 µs RMS in every
  session — the same seed draws the same 1200 values, whose spread is that much below the configured one (sampling,
  about 2 % per standard error).

An earlier attempt at the same sessions ran during a load peak of this machine (load average 60–75, other projects'
test suites): NVENC ran at 0.53× real time, x264 at 0.24× with no control on time, and in the raw session the CARLA
server stopped answering after 1185 steps (the watchdog ended it, exit 4). Those numbers are not used above; they show
that a shared machine cannot hold a 20 Hz loop with this rig, and that the session says so (the real-time factor,
steps late, the load average in the report) instead of hiding it.

**With the fake CARLA** (the test suite: 192 × 108 at 10 Hz, a 16-channel lidar), all three transports close the loop
with every frame decoded; the step-to-control latency there is dominated by the fake's own Python rendering.

**Pending on the Jetson AGX Orin** — nothing above was measured on it; the acceptance run will measure:

- the real network (both one-way stages, their jitter, loss over UDP; the offset estimate's uncertainty on a LAN);
- `nvv4l2decoder` + `nvvidconv` decoding at 1280 × 720 (here `avdec_h264` on a desktop CPU);
- the stack's time on the Orin's CPU (the default stack is trivial; a real stack's time is the point of the exercise);
- the raw transport's wire time at 442 Mbit/s on gigabit Ethernet;
- JetPack 6's GStreamer 1.20 with the agent (here it ran under Python 3.10 and numpy 1.21 in the
  `ros:humble-ros-base` container, without GStreamer; the H.264 path ran only under GStreamer 1.28 on this machine);
- the ROS 2 republishing on the Orin's DDS (here only inside the container).

What you need to provide for it: the Orin's address on the LAN, an SSH user with your public key installed
(`ssh-copy-id`), the key's path on this machine, inbound TCP/UDP on ports 7710–7712 (or the port you choose and the two
after it), and on the Orin `python3-numpy`, `python3-gi`, `gir1.2-gstreamer-1.0` and `gir1.2-gst-plugins-base-1.0`
(usually present on JetPack 6). Then `falcon hil acceptance PROJECT` (or *Run the acceptance*).

## CAN

`--can IF` (`falcon hil session`, or *CAN* on the Hardware page) puts a **chassis model** (below) on a CAN bus: the
ECUs a device talks to. The device's stack sends its control as CAN command frames; the chassis's actuators follow
them, and with control `can` (the default) **the ego is driven by what the actuators do** — the session's own control
message is still recorded (its stamps), not applied. With control `record` CAN runs beside the session's control. The
chassis's feedback — wheel ticks and speeds, yaw rate, steering, brake pressure — goes out every 10 ms, and the device
agent hands the newest to the stack (`meta["can"]`).

Interfaces: a SocketCAN interface (`vcan0`; a USB-CAN / CAN FD adapter with a SocketCAN driver such as gs_usb,
peak_usb or kvaser_usb as `can0`; the Jetson's own `mttcan`), opened with the standard library's `AF_CAN` /
`CAN_RAW` (python-can is not needed); `sim:NAME`, a **simulated link** between the processes on this machine
(abstract Unix datagram sockets: no vcan, no privileges, nothing on the network) — what the simulated device uses when
there is no vcan; `python-can:INTERFACE:CHANNEL` for an adapter without a SocketCAN driver (python-can optional, the
`hil` extra; its timestamps are the adapter's). CAN FD frames only when the interface's MTU is 72 (`fd auto`, the
default; `fd on` refuses a classical interface). The device side picks its own interface on its command line
(`device_agent.py --can can0`, or `falcon hil deploy --start --agent-can can0`): the host never chooses the device's
hardware.

**The messages** (`core/canmsgs.py`; `falcon hil can dbc` writes `resources/can/falcon_chassis.dbc`, which
cantools, SavvyCAN and CANdb++ read). 11-bit identifiers, little-endian signals; every message ends with a 4-bit
alive counter (the low nibble of its second-last byte) and a CRC-8 SAE J1850 (polynomial 0x1D, initial and final
XOR 0xFF — AUTOSAR E2E profile 1's) over the identifier's two bytes and the bytes before it, so a frame copied onto
another identifier fails.

| id | message | bytes | every | from | signals |
|---|---|---|---|---|---|
| 0x0A0 | WHEEL_TICKS | 8 | 10 ms | chassis | TICKS_FL/FR/RL/RR (12 bits each, wrap at 4096), QUALIFIER |
| 0x0A1 | WHEEL_SPEEDS | 8 | 10 ms | chassis | SPEED_FL/FR/RL/RR (12 bits, 0.0625 km/h) |
| 0x0A2 | YAW_ACCEL | 8 | 10 ms | chassis | YAW_RATE (0.01 °/s, left +), LAT_ACCEL, LONG_ACCEL (0.001 m/s²), STATUS |
| 0x0A3 | STEERING | 8 | 10 ms | chassis | STEER_ANGLE (0.1 °, steering wheel, left +), STEER_RATE (°/s), EPS_TORQUE (0.01 N·m), EPS_STATUS |
| 0x0A4 | BRAKE | 8 | 10 ms | chassis | BRAKE_PRESSURE, BRAKE_REQ_ECHO (0.01 bar), VEHICLE_SPEED (0.01 km/h), BRAKE_STATUS |
| 0x0A5 | CHASSIS_STATUS | 8 | 100 ms | chassis | MODE (idle, active, safe_hold, aborted), SURFACE, MU_EFF, WATCHDOG_TRIPS, CMD_REJECTED, HYDROPLANING |
| 0x0B0 | CMD_STEER | 8 | 20 ms | device | STEER_REQ (0.1 °), STEER_TORQUE_REQ (0.01 N·m), STEER_MODE (off, angle, torque), ENABLE, STEP (12 bits) |
| 0x0B1 | CMD_LONG | 8 | 20 ms | device | BRAKE_REQ (0.01 bar), THROTTLE_REQ (0.5 %), ENABLE, STEP (16 bits) |
| 0x0C0 | CHASSIS_FD | 64 (FD, BRS) | 10 ms | chassis | all the feedback in one frame, ticks 16 bits, SIM_TIME |
| 0x0C1 | CMD_FD | 64 (FD, BRS) | 20 ms | device | the steering and longitudinal command in one frame, STEP 32 bits |

On CAN FD the bridge sends CHASSIS_FD (and the status) instead of the five classical feedback frames; a device may
command with CMD_FD or the classical pair. STEP is the simulation step the command answers: it ties a command to its
frame's latency record. The device agent maps its stack's output: `steer` (CARLA's, right positive) × −steer_max_deg
(540°) → STEER_REQ, `brake` × brake_full_bar (100 bar) → BRAKE_REQ, `throttle` × 100 → THROTTLE_REQ; a stack may
return `steer_torque_nm` (torque mode) or `brake_bar` directly. It sends a step's command at once, then repeats the
newest every 20 ms with fresh counters.

**A DBC of your own** (`--dbc FILE --dbc-map ROLE=MESSAGE.SIGNAL …`, or the DBC and signal map on the page): the
bridge then speaks your messages — each feedback role (TICKS_FL … VEHICLE_SPEED) encoded into its message (the
message's other signals 0; an alive counter only if you map `COUNTER=MESSAGE.SIGNAL`: a DBC does not say how a vehicle
protects its frames, so no CRC is computed), each command role (STEER_REQ, BRAKE_REQ, THROTTLE_REQ, STEER_MODE,
ENABLE, STEP) decoded from its. Units are not converted: the check says when a mapped signal's unit is not the role's.
cantools (optional) reads the DBC when installed (multiplexing, every attribute); without it Falcon's own reader takes
BO_/SG_ lines in both byte orders and skips multiplexed signals, saying so.

**Safety interlocks** — CAN on a real vehicle can move it; Falcon's are for a bench:

- **The allow-list**: Falcon opens only interfaces matching settings `hil_can.interfaces_allowed` (default `vcan*`;
  Settings ▸ Hardware). A real adapter (any interface not under `/sys/devices/virtual`) also needs **a confirmation
  that it is wired to a bench network, not a vehicle** — typed as exactly that ("bench network, not a vehicle") in
  Settings ▸ Hardware, on the Hardware page or with `falcon hil can confirm can0 --statement "…"` — recorded with its
  time, per adapter, withdrawn with `falcon hil can revoke`. Patterns alone never allow a real adapter. The worker
  checks again against your settings, so a config edited by hand is refused too (exit "refused").
- **The command watchdog**: a command is valid when its CRC checks, its alive counter moved on by 1–3 and its values
  are in range. With none for **100 ms** (both the steering and the longitudinal part), the chassis goes to a **safe
  hold**: no throttle, the steering held where it was, the brake at `safe_brake_bar` (40 bar; a held stop once
  stationary); counted in WATCHDOG_TRIPS. Control comes back only after **three valid commands in a row**. The same
  hold before the first command and while a command says ENABLE 0 (not counted as a trip). Rejected frames are
  counted (CMD_REJECTED) and never feed the watchdog.
- **Bus errors abort**: an error frame (bus-off, error-passive, no acknowledgement, a protocol violation …; error
  frames are subscribed with `CAN_RAW_ERR_FILTER`) or 50 failed sends in a row ends the session with the ego held and
  the reason in run.json — a sick bus is not a test.
- **Silent on exit**: the bridge stops sending before it closes its socket; it uses raw sockets only (no broadcast
  manager jobs that would outlive it), so nothing is left transmitting after a session — finished, failed or
  cancelled.

**The "can" stage.** With CAN control the waterfall's way back is: the stack's output onto the bus (*Control onto
the CAN bus*, device), the bus to this machine (*CAN bus to this machine*, one-way: the device's send stamp against
the kernel's receive timestamp) and **CAN frame to the simulation** — from the kernel's receive timestamp of the
first frame carrying the step's command (`SO_TIMESTAMPNS`, CLOCK_REALTIME, put on CLOCK_MONOTONIC by the offset
between the two read at that moment) to the bridge's read in the worker — then the wait for the next step. The
chassis's own actuator delays and lags are the modelled vehicle's, not latency stages: they are in its parameters.

`falcon hil can bench IF` runs the chassis model alone on a bus (its own vehicle body as the plant) for a device on the
bench, printing its mode and counters; `falcon hil can interfaces` lists what is here and whether Falcon may open it.

## The chassis model

`core/chassis.py` (numpy; it runs in the session worker in CARLA's Python). Every number is a parameter
(`--surface`, `--mu`, or the config's `can.chassis`); the defaults are typical of a mid-size car, chosen, not
measured:

| part | model | defaults |
|---|---|---|
| steering (EPS) | the steering wheel angle follows its target with a first-order lag after a pure transport delay (a ring buffer), rate-limited; angle mode: the target is the request; torque mode: the angle where the requested torque balances a linear self-aligning spring | τ 80 ms, delay 30 ms, 720 °/s, 60 °/N·m, ±540°, ratio 15 |
| brake | a dead band (backlash: the pressure's target moves only when the request leaves ± half the band; below half a band it releases fully), then different apply and release rates, after a transport delay | band 2 bar, apply 300 bar/s, release 150 bar/s, delay 20 ms, 0–150 bar, 14 / 8 N·m per bar front / rear |
| wheels | J·dω/dt = T_drive − T_brake − F_x·r (the brake never turns a wheel backwards; a spinning wheel is capped at a slip of 1, a traction control's limit); slip ratio with a relaxation length, σ·dκ/dt + max(\|v\|, 0.5)·κ = ω·r − v (implicit, stable at standstill); F_x = μ_eff·F_z·Pacejka's magic formula (B 10, C 1.9, E 0.97); F_z with longitudinal load transfer | 1500 kg, wheelbase 2.7 m, r 0.31 m, J 1.2 kg·m², front drive, 3000 N·m |
| ticks | the wheel's angle × teeth / 2π, counted from the start: a locked wheel stops ticking while the car slides | 48 teeth |
| surface | μ per surface (typical peak values): dry asphalt 0.9 (0.8–1.0), wet 0.6 (0.5–0.7), **waterlogged 0.4 (0.3–0.5)**, gravel 0.55, snow 0.25, ice 0.1 | |
| hydroplaning | waterlogged only: **Horne's dynamic hydroplaning speed v_p ≈ 6.35·√p km/h** (p the tyre pressure, kPa; NASA TN D-2056: 9·√psi knots); μ_eff falls linearly from μ at 0.7·v_p to 0.05 at v_p and stays there — the ramp is a heuristic, Horne's speed the only measured anchor | 220 kPa → 94.2 km/h |
| body | plant `model` (the bench): m·dv/dt = ΣF_x − drag − rolling resistance, the yaw rate of a kinematic bicycle capped at μ_eff·g lateral; plant `carla` (a session): the motion is CARLA's | |

**Measured by the tests** (`tests/test_chassis.py`): a 30° steering step moves nothing for 30 ms and reaches 63 % at
110 ms (delay + τ, ±2 ms); a slow brake sweep up and down draws a loop exactly the dead band wide (4.00 bar for a
4 bar band at every level; a 20 ms delay at 10 bar/s widens it by 0.4 bar, as it should); 0 → 90 bar takes 0.30 s
and back 0.60 s. Braking at 40 bar from 100 km/h: on a dry road the slip stays under 10 % and the ticks follow the
distance travelled within 5 %; on a waterlogged one (past the 94 km/h hydroplaning speed) the wheels lock (slip below
−0.9), the car hardly slows and the wheels tick less than half as often as the distance would say.

**With CARLA as the plant** the vehicle's motion is CARLA's own tyre model; the wheels here are driven by that motion
(speed, yaw rate, the acceleration over each step) and the surface's μ — the brake by the modelled pressure, the drive
by the torque that produced CARLA's acceleration (m·a·r while the throttle is on, so a wheel spins only where the
surface could not carry it) — and on a slippery surface their slip and CARLA's deceleration agree only as far as
CARLA's friction matches: a session scales the ego's `tire_friction` by μ / 0.9 when CARLA exposes it (recorded
in run.json `can_vehicle`); the hydroplaning ramp acts on the wheel signals only. The actuators map onto CARLA's
control: the road wheel angle over the vehicle's maximum steer angle (its front wheels' `max_steer_angle`), the
pressure over 100 bar, the throttle as is.

## The camera as a V4L2 device

`--camera v4l2` (*A V4L2 device on this machine* on the page) writes each frame into a **V4L2 output device** — a
v4l2loopback node (`/dev/video40`, `falcon hil setup` prints how to create it) — which any program then opens as an
ordinary camera: a ROS 2 camera driver, GStreamer's `v4l2src`, a perception stack under test. The writer is the
standard library's (`hil/v4l2out.py`): VIDIOC_QUERYCAP (it must be an output device), VIDIOC_S_FMT with the size and
YUYV, UYVY, RGB3 or NV12 (BT.601 limited range, what a USB or GMSL camera's YUV usually is; the driver's answer is
checked), then one `write()` per frame; or GStreamer's `v4l2sink` (`--v4l2-method gstreamer`). Each frame's stamps
(`{"f": frame, "s": sim_t, "h": hw_t, "w": write start}`) go on a UDP side stream to the agent's port P + 3, and with
the **barcode** (on by default) the frame number is painted into the top two rows — 44 cells of white or black,
grey so that no chroma is touched, surviving 4:2:2 and 4:2:0 — so that whatever reads the device can match each
frame to its step and measure the path's latency. The barcode is part of the image the stack sees.

The simulated device reads the same node back as a camera (`--camera-device`, set by the gateway) through
`hil/gmsl.py`'s capture grabber, matching frames by their barcode and putting the driver's buffer timestamp on its
clock: the waterfall's *Network to the device* is then write() → the buffer's timestamp, and *Decode* the conversion
to BGR. A remote device reads its own capture node (`device_agent.py --camera-device /dev/videoN`).

## GMSL2

On the Jetson AGX Orin a GMSL2 camera appears through its deserialiser (MAX9296 / MAX96712 boards and the Jetson's VI
driver) as `/dev/videoN`. `hil/gmsl.py`'s **V4L2CaptureGrabber** reads such a node by memory-mapped streaming I/O
(REQBUFS, QUERYBUF + mmap, QBUF, STREAMON, select, DQBUF) and reports each frame's driver sequence number and
timestamp with the clock and source its flags state (CLOCK_MONOTONIC, start of exposure or end of frame) — never
assumed. It reads YUYV, UYVY, RGB3, BGR3, NV12 and GREY single-plane nodes; raw Bayer through VI (what Argus's ISP
would process) is not demosaiced, and multi-planar nodes are refused.

**Injecting simulated frames into a GMSL2 link is not possible here**: it needs serializer-injection hardware (a
frame injector feeding a MAX9295 serializer, for example), which this project does not have. `hil/gmsl.py` defines
the `Injector` interface and only a `FakeInjector` (an in-memory queue with a fixed latency, read back through its
own grabber) — enough to test the code paths, nothing of GMSL2's protocol. With a real injector the device's own
capture path (deserialiser, VI, driver) would carry Falcon's frames; what a GMSL2 path adds and Falcon does not model:
the serdes latency (fixed per link, sub-millisecond), exposure and rolling shutter (the session stamps the
simulation time), the ISP.

## Clocks: PTP and the NTP-style fallback

The offset between this machine's and the device's clock has two sources (`core/ptp.py`); a session computes
**both** when it can and records them with their **disagreement**, and settings `hil_ptp.clock_source` (or
`--clock-source`, the page's *Clock source*) says which the waterfall uses:

- **ntp-style** — the UDP pings above (always running; the fallback);
- **ptp** — the device reports, once a second, its CLOCK_REALTIME and its stamp clock read together, and every 5 s
  what its `pmc` says (raw: the host parses it); the gateway notes this machine's CLOCK_REALTIME − CLOCK_MONOTONIC
  when each report arrives and its own `pmc` every 5 s (`ptp.jsonl`). If both realtime clocks follow one grandmaster,
  θ = (host realtime − monotonic) − (device realtime − stamp clock), fitted with a line as the pings are; its
  uncertainty is the fit's residual + the worst |master offset| either `pmc` reported + 2 µs for two clock reads.
  PTP counts only when both `pmc` report a grandmaster present, **the same** grandmaster, a SLAVE or MASTER port and
  |offset| < 1 ms; otherwise the reasons are recorded and the ping estimate is used. A simulated device shares this
  machine's CLOCK_REALTIME by construction — what PTP provides between two machines — so its estimate is exact up to
  the clock reads and exercises the same path (basis "shared realtime");
- **auto** (default) — PTP when it may be used, else the pings.

**Software versus hardware timestamping, honestly.** This machine's NICs (and most desktop and laptop NICs) have **no
PTP hardware clock**: `ethtool -T` lists only software timestamping (a Wi-Fi interface often not even transmit
timestamps — ptp4l cannot run on it). With **software timestamping** ptp4l stamps packets in the kernel's network
stack, after interrupt and softirq latency, and steers CLOCK_REALTIME directly (no phc2sys): expect **tens of
microseconds** of error, rising under load — about what the NTP-style estimate already gives over a quiet LAN, so on
such hardware PTP mostly confirms the pings rather than beating them. **Hardware timestamping** (a NIC with a PHC:
`hardware-transmit`, `hardware-receive`, a "PTP Hardware Clock") stamps at the MAC/PHY and reaches sub-microsecond
agreement; phc2sys must then bring CLOCK_REALTIME to the PHC, and its own offset joins the error. Whether the Orin's
Ethernet offers hardware timestamping, `ethtool -T` there says (its `--check` reports pmc and ptp4l, `falcon hil
deploy` shows it); this machine's NIC does not — so an Orin-to-host link is at best software-timestamped at this end,
and the uncertainty says so.

`falcon hil ptp conf --profile software|gptp [--interface IF] [--timestamping software|hardware]` prints a
`ptp4l.conf` to install: **software** — the default E2E/UDPv4 profile with `time_stamping software`; **gptp** —
linuxptp's gPTP (IEEE 802.1AS) profile: `transportSpecific 0x1`, Layer 2 to 01:80:C2:00:00:0E, peer-to-peer delay,
two-step, `follow_up_info`, hardware timestamping. 802.1AS requires a neighbour delay under 800 ns to be
"asCapable", which software stamps cannot measure: with `--timestamping software` the file raises the threshold and
says that the result is **not a conformant gPTP setup** (for testing the plumbing only). `falcon hil ptp status`
shows ptp4l / phc2sys / pmc, `ethtool -T` and the `pmc` answer here. The parsers (`pmc` TIME_STATUS_NP,
PORT_DATA_SET, CURRENT_DATA_SET; `ptp4l -m` / `phc2sys -m` lines; `ethtool -T` in its old and new wording) are tested
against recorded output and stand-in programs.

## Setting up the interfaces

Every interface needs root once, and **Falcon never runs anything with privileges**: `falcon hil setup [--can-interface
can0] [--ptp-interface IF]` **prints** the commands for you to run — vcan (`modprobe vcan`, `ip link add dev vcan0 type
vcan`, `mtu 72` for CAN FD), a USB-CAN / CAN FD adapter (`ip link set can0 up type can bitrate 500000 dbitrate 2000000
fd on`) and its bench confirmation, the Orin's mttcan, v4l2loopback (`modprobe v4l2loopback devices=2
video_nr=40,41 card_label=falcon exclusive_caps=1`) and PTP (linuxptp, the generated file, `ptp4l -f … -m`), and how
to undo them. `falcon doctor` lists what is in place (informational: none of it is required).

## Measured here, and what needs the hardware

The CAN loop, the clock sources and the V4L2 paths were measured **with the fake carla and the simulated device on
a simulated CAN link** (this machine has no vcan, no v4l2loopback and no PTP, and Falcon creates none):

30 s at 10 Hz, the camera 192 × 108 as raw BGR over TCP, a 16-channel lidar, control `can`, wet asphalt; the
simulated device with an emulated clock (offset 1500 ms, 20 ppm); the machine shared (1-minute load average 15–17).
2026-11-12. Median / p95 in ms:

| stage | classical CAN | CAN FD |
|---|---|---|
| Simulation step (the fake carla's Python rendering) | 33.3 / 70.5 | 34.1 / 60.1 |
| Stack (brake_for_obstacles) | 0.18 / 0.39 | 0.19 / 0.40 |
| Control onto the CAN bus (device) | 0.054 / 0.126 | 0.072 / 0.138 |
| CAN bus to this machine (one-way, simulated link) | 0.033 / 0.103 | 0.015 / 0.035 |
| **CAN frame to the simulation** (kernel timestamp → the bridge's read) | **0.079 / 0.94** | **0.040 / 0.58** |
| Step start to control at the simulation | 35.2 / 72.9 | 35.7 / 63.2 |
| End to end (one step at 10 Hz) | 99.8 / 103.0 | 99.8 / 102.5 |
| On time / superseded (of 299) | 294 / 5 | 297 / 2 |
| Frames sent by the bridge / commands received | 16 028 / 3 561 | 3 476 / 1 779 |
| Bus load (nominal bits at 500 kbit/s) | 11.1 % | — |
| Rejected frames, error frames, saturated signals | 0, 0, 0 | 0, 0, 0 |
| Watchdog trips | 1 | 0 |
| Clock offset used: PTP-style (shared realtime) ± | 3.8 µs; off the known truth by 0.3 µs | 4.5 µs; off by 1.0 µs |
| NTP-style pings ± | 44.6 µs; off by 12.7 µs | 47.4 µs; off by 10.5 µs |
| Their disagreement | 13.1 µs (consistent) | 9.5 µs (consistent) |

What these say: the "can" stage is the host's own receive path — a fraction of a millisecond at the median, about one
at p95 on a loaded machine — and a real adapter adds its USB latency in front of it (not measured here). The simulated
link has no wire time; a classical frame of 8 bytes is about 0.22 ms on a 500 kbit/s bus, which a real bus will add
to *CAN bus to this machine*. The realtime-based estimate (a simulated device: shared CLOCK_REALTIME) is an order of
magnitude tighter than the pings on loopback, and both contained the known truth; across two machines it is only as
good as PTP's own offset. The classical session's one watchdog trip: the device's commands paused for over 100 ms
(on a machine this loaded, or as the session wound down) and the chassis held the vehicle, as it should; `can.json`
records each trip's time since (`watchdog_trip_s`).

What needs the user's hardware or a `sudo` once to confirm (each has an opt-in test):

- **vcan** (`FALCON_VCAN=1`, `tests/test_hil_can.py::test_can_session_on_vcan0`): the same session over SocketCAN —
  the kernel's timestamps from a real `AF_CAN` socket, CAN FD at MTU 72;
- **the USB-CAN/CAN FD adapter and the Orin**: the device agent on the Orin with `--can can0` (mttcan, a transceiver
  on its header) and the adapter on this machine confirmed as a bench network; bus load, error frames on a real bus
  (no acknowledgement when the other node is off), the "can" stage through a USB adapter's own latency (gs_usb and
  others batch frames: expect a millisecond or more), `restart-ms` after a bus-off;
- **v4l2loopback** (`FALCON_V4L2=1`, `tests/test_v4l2.py::test_real_v4l2loopback_write_and_capture`): the writer and
  the grabber against the kernel module, the write → buffer-timestamp latency;
- **GMSL2 on the Orin** (`FALCON_ORIN=1`, `FALCON_GMSL_DEVICE=/dev/videoN`): driver timestamps from a real
  deserialiser; injection needs hardware Falcon does not have;
- **PTP**: ptp4l on both machines (software timestamping at this machine's end), the session's two clock sources and
  their disagreement over the real LAN.

## Limits

- **One camera and one lidar** per session (the rig's first pinhole camera and first lidar).
- The camera over ROS 2 DDS is not offered here (the stack-testing bridge publishes images); the lidar is, as
  PointCloud2. Raw video over RTP (RFC 4175) is not offered: raw goes over TCP.
- No jitter buffer: a lost RTP packet loses its frame (counted); a stack then runs on the lidar alone after one step.
- The emulated sensor clocks stamp the **simulation time** of the step; they do not model exposure or a rolling shutter
  (the rig's faults do that for recorded data).
- One-way network stages rest on the symmetric-delay assumption (see above); the round trip does not.
- The hand-off from CARLA's Python to the gateway is a local copy of every frame (CARLA's venv has no GStreamer); it is
  measured and part of the waterfall.
- The simulation is paced to real time and never waits for the device; a session whose step takes longer than its
  period runs slower than real time (the real-time factor says so) and its deadlines are counted against the step.
- The streams are neither encrypted nor authenticated: use a trusted network.
- CAN: the chassis is a model, not a vehicle's ECUs — no gateway, no diagnostics, no network management; its CRC and
  counter scheme is Falcon's (a user DBC's protection is not computed). The simulated link is point to point and
  has no acknowledgement: a frame the other end does not read is lost silently, where a real bus would raise errors.
- V4L2: one camera, single-plane formats, the 64-bit kernel ABI (x86-64, aarch64); the barcode is in the image.
- GMSL2: capture only; injection needs hardware Falcon does not have (FakeInjector only).
- PTP: Falcon reads linuxptp, it does not run it; with software timestamping the realtime estimate is no better
  than tens of microseconds.
