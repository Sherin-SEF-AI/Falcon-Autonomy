# Lessons from CARLA 0.9.16

What Falcon does because of how CARLA 0.9.16 behaves (measured on the reference machine unless stated).

- **Signals and threads.** numpy's PyPI OpenBLAS starts about 23 threads at import. Workers block SIGINT and SIGTERM
  before importing numpy and take them in a watchdog thread with `sigtimedwait`; otherwise a cancel can kill the
  worker without tidying the server.
- **A crashed server hangs the client** forever in its call. The watchdog ends a worker whose server stopped
  answering, with exit code 4 and `run.json` saying why.
- **Tidy-up order.** Destroying the ego before the sensors attached to it, stepping the world after destroying
  vehicles the traffic manager drove, or calling `Actor.destroy()` on any vehicle that was ever on the autopilot (even
  after `set_autopilot(False)`) makes libcarla throw "trying to operate on a destroyed actor" in its own thread and end
  the process with `std::terminate`. Falcon stops the sensors, takes the vehicles off the autopilot, steps once,
  destroys the sensors with `destroy()` and everything else **by id in a batch** (`apply_batch_sync` with
  `command.DestroyActor`), then puts the world back in asynchronous mode without another synchronous step.
- **A fresh client of a synchronous server sees no actors** until a step is taken (no snapshot reaches it): tidying
  after a killed worker steps once first.
- **Asking for a traffic manager starts one** in the asking process when none runs on that port: the tidy worker never
  asks for one.
- **`ActorAttribute` has no `value`** in the wheel: use `as_str()`, `as_int()`, `as_float()`, `as_bool()`.
- **Synchronous mode with a fixed step.** Warm up a few steps before recording (the first frames after a spawn are
  not settled), and always restore asynchronous mode on exit.
- **The Vulkan driver.** Without `VK_ICD_FILENAMES` pinned to NVIDIA's, Unreal can pick the integrated GPU.
- **Memory.** A server capped at 12 GB crashed at 306 Mpx rendered per step; one server plus a heavy GUI job needs
  about 22 GB free.
- **Instance segmentation**: R = the semantic tag, G + 256·B = the actor id for actors; map meshes (parked cars,
  signs, buildings) carry unrelated ids. Traffic-light heads and signs are better claimed by their boxes than their
  pixels. *(Used from milestone 3.)*
- **Depth is planar** (along the view axis): convert it to range along each pixel's ray; interpolate inverse depth
  only within one surface. *(Milestone 3.)*
- **The semantic lidar** casts the same rays as the ray-cast lidar on the same tick: match points by order, and prove
  it with a range check. *(Milestone 3.)*
- **Weather**: lidar and radar ignore it; rain is water on the lens, not loss of range; fog visibility ≈ 7000 m /
  fog_density.
- **CARLA's own car is not your vehicle body**: body masks and drift checks use CARLA's car, from the instance id.
  *(Milestone 3.)*
- **Fisheye from cube faces** needs ~1.5° of overlap past the seams with feathering, and Unreal's vignette removed in
  linear light through CARLA's camera response. *(Milestone 3.)*
- **OpenStreetMap conversion (`carla.Osm2Odr`)** writes right-hand roads only (no handedness option; no `rule`),
  projects with `+proj=tmerc` centred on 0°E unless told otherwise (central Bengaluru came out stretched about 3× by
  4×), and writes some crosswalk objects a few millimetres past their road's end — which makes
  `generate_opendrive_world` crash the server (signal 11, *s <= road->GetLength()*).
- **`rule="LHT"` is honoured** by CARLA 0.9.16's map (waypoint directions) and traffic manager (driving side).
- **Generated OpenDRIVE worlds label their road and sidewalk meshes as sky (11)** in the semantic camera.
- **The server removes traffic-manager vehicles** that reach a dead end (a cut area has many): a client then gets
  "trying to operate on a destroyed actor" from that vehicle's methods. Check `is_alive` and catch it.
- **Python's own modules can be shadowed** by a script's folder: a helper named `bisect.py` beside a script broke
  `urllib.request`. Name scratch scripts after what they do, not after a word that is also a module.

## Native ROS 2 (measured, milestone 7)

Measured on 2026-10-28 on Falcon's own server started with `--ros2` (the setting `server_ros2`; port 2340, an 8 GB
cap, Town10HD_Opt, synchronous mode at 0.05 s), subscribed from the official `ros:humble-ros-base` image (with the
control message packages added: `falcon_autonomy/ros2/Dockerfile`) on the host network, rmw_fastrtps_cpp, Fast DDS on
UDP only, ROS domain 0. The ego (`vehicle.lincoln.mkz_2017`, `ros_name` "hero") carried a 1280 × 720 RGB camera, a
32-channel lidar (600 000 points/s, 20 Hz), an IMU and a GNSS, each with a `ros_name` and `enable_for_ros()`.

- **Topics.** Only sensors enabled for ROS publish, under `/carla/<vehicle ros_name>/<sensor ros_name>/…`:
  `…/front/image` (`sensor_msgs/Image`, **bgra8**, 1280 × 720, step 5120), `…/front/camera_info`,
  `…/lidar/point_cloud` (`PointCloud2`: x y z intensity float32, point_step 16, about 15 000 points a message for these
  settings, is_dense false), `…/imu` (`Imu`), `…/gnss` (`NavSatFix`), plus `/clock` and `/tf`. Each header's frame_id
  is the sensor's `ros_name`. `/tf` holds only the mounts (hero → each sensor), in ROS's axes: a sensor mounted at
  CARLA y = +0.8 m, yaw 30° came out at y = −0.8 m, yaw −30°. **No odometry, no map → hero transform, no ego pose and
  no vehicle status** are published: a stack has to localise from GNSS and IMU or get its pose elsewhere.
- **Rate in synchronous mode:** one message per enabled sensor per tick (camera_info, IMU, GNSS and lidar counted equal
  to the ticks). With a best-effort subscriber, only 70–81 % of the 3.7 MB images arrived (211 of 300 ticks as fast as
  possible, 162 of 200 at 20 Hz); with a reliable subscriber all of them did. `/clock` is published once per tick with
  the tick's simulation time, and continuously (about every server frame) in asynchronous mode.
- **Stamps:** every sensor header stamp was the World snapshot's `elapsed_seconds` of its tick **plus one fixed
  step** (0.05 s here, for every IMU message), i.e. one step later than the `/clock` of the same tick.
- **camera_info is wrong:** K had fx = fy = −36 757.5 for a 1280 × 720 camera with a 90° field of view (640 expected);
  cx = 640, cy = 360, plumb_bob with zero distortion. Compute K from the blueprint instead.
- **IMU:** the first message after the spawn read 25 630 m/s² (the spawn), then settled (median z 9.81 m/s²).
- **Control is accepted, in synchronous mode.** CARLA subscribes `/carla/hero/vehicle_control_cmd`
  (`carla_msgs/CarlaEgoVehicleControl`) and `/carla/hero/ackermann_control_cmd` (`ackermann_msgs/AckermannDriveStamped`).
  Throttle 0.6 published at 50 Hz took the ego from rest to 8.9 m/s in 5 s of simulation (get_control() read throttle
  0.6); an Ackermann command of 5 m/s took it to 4.35 m/s in 5 s (CARLA's own controller: throttle 0.42). Both were
  applied on the client's ticks. Not measured: whether CARLA uses the header stamps, what it does with both topics at
  once, its latency. `carla_msgs` is not released for ROS 2 Humble: the image builds `CarlaEgoVehicleControl.msg` itself.
- **What it does not do:** CARLA does not wait for a stack. The ticking client advances the world whenever it likes,
  so a closed loop with a deadline per step needs another component — Falcon's bridge ([stack testing](stack-testing.md)).
- **The official carla-ros-bridge does not support 0.9.16** (read, not run): its master branch's
  `carla_ros_bridge/src/carla_ros_bridge/CARLA_VERSION` says 0.9.13 and `bridge.py` exits ("CARLA python module version
  0.9.13 required") when the installed `carla` package is another version; its last commit is of 2022-07-22 and its
  last release 0.9.12. It also imports `distutils`, which Python 3.12 (CARLA 0.9.16's client here) no longer has;
  CARLA 0.9.16 publishes cp310 wheels, so Humble's Python 3.10 could import it, but the version check refuses it.
- **rclpy's CDR** (ROS 2 Humble's `serialize_message`): alignment padding bytes are not initialised (a 0x76 seen after
  a bool), and a message is sometimes ended with up to three more zero bytes (a 13-byte GearReport is 16 bytes, a
  78-byte Control is 78). Compare messages field by field, not byte for byte.
- **Memory between servers:** a 0.9.16 server at 5.6 GB, well under its 8 GB scope cap, was killed by the kernel's OOM
  killer while another process's 6.4 GB CARLA server ran on the machine. The memory guard checks free memory when a
  server starts, not later. A crashed scope stays loaded as failed under its unit name, so a restart in the same
  process needs a new unit name (sim/server.py now numbers its restarts).
