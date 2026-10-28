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
