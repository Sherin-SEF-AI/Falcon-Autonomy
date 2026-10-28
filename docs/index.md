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

- **Navigator** (left): the pages. Ctrl+1 … Ctrl+9 go to them.
- **Jobs** (bottom): every long action — queued, waiting (and why), running with its progress and ETA, finished.
  The ✕ cancels a job; a cancelled worker tidies the server first.
- **Log** (bottom): what every job and the app said.
- **Status bar**: Falcon's CARLA server (start or stop it there), the GPU's memory and load, free memory.
- **Command palette** (Ctrl+K): every command and page by name; type a few letters.
- **Undo and redo** (Ctrl+Z, Ctrl+Shift+Z): every change to the project. Dragging a slider is one step.
- **Themes**: View ▸ Dark theme / Light theme.

## Pages

### Home
The server (state, port, folder, client Python, the last probe), the machine's checks, the throughput Falcon uses
for estimates and where it came from, and the latest runs (double-click one to browse it).

### Worlds
A world is a CARLA town with its weather and time of day, and the traffic profile it drives with. Presets set all
of CARLA's weather fields; sliders override them one by one. *Reads as* says what the weather amounts to (day or
night, rain, fog visibility).

Limits, stated where weather is chosen: **CARLA's lidar and radar ignore weather** (no loss of range in rain or
fog); rain on a camera is water on the lens and wet surfaces, not loss of range; **fog visibility ≈ 7000 m / fog
density** (measured on 0.9.16, a 5 % contrast threshold — a human observer's; for a detector it is an upper bound).

The town list is the base package's (Town01–05, Town10HD and their `_Opt` variants) until Tools ▸ Probe the CARLA
server lists the server's own maps.

### Data
**Record** a drive (see *Getting started*). **Browse** recorded runs: choose the camera and the layer, scrub the
frames, set the overlay's opacity. A cancelled or failed run keeps only `run.json` (why it ended) and its input,
never half a dataset.

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
