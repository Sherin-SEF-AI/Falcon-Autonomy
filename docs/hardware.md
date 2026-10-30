# Hardware in the loop

Hardware ▸ Session (or `falcon hil`) streams one camera and one lidar of a CARLA session to a device in real time — a
Jetson AGX Orin on the network, or a **simulated device** on this machine — runs a stack there, takes its control back
and applies it to the ego at the next simulation step. Every frame's latency is recorded stage by stage, from the
step that rendered it to the step that applied the control computed from it, and shown as a **latency waterfall**.

The code: `core/hil.py` (configs, emulated sensor clocks, the clock-offset estimate, the statistics),
`workers/hil.py` (the session worker), `hil/gateway.py` (the streams), `hil/device_agent.py` (the device side, one
file), `hil/report.py` (the report), `hil/deploy.py` (SSH), `hil/cli.py` (`falcon hil`), `ui/pages/hardware.py`.

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
