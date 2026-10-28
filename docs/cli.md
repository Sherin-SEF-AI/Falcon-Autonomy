# Command line

`falcon` (or `python3 -m falcon_autonomy`) runs every heavy action headless, on the same code paths as the GUI: the
same configs, job queue (with the machine-wide CARLA lock and memory guard), workers and protocol.

| command | what it does |
|---|---|
| `falcon [gui] [PROJECT]` | the application |
| `falcon doctor` | checks CARLA, its Python, the GPU, the Vulkan driver, memory, systemd-run and Falcon's ports |
| `falcon project new NAME PATH` | a new project file |
| `falcon project validate PATH` | checks a project against its schema (migrating in memory) |
| `falcon project migrate PATH` | migrates a project to this build's format, in place |
| `falcon probe [--wait S]` | what server answers on Falcon's port (starts Falcon's server if none) |
| `falcon smoke PROJECT` | the 30 s smoke drive; records this machine's throughput |
| `falcon drive PROJECT [--world ID] [--rig ID] [--seconds S] [--fps F] [--seed N] [--out DIR] [--max-wall S]` | a recorded drive |
| `falcon tidy` | tidies Falcon's server after a killed worker |
| `falcon reproduce RUN [--into DIR] [--pose-tol M] [--rgb-tol L]` | re-runs a run from its input and compares |
| `falcon runs PROJECT [--rebuild]` | lists a project's runs (re-indexing the workspace) |
| `falcon schema` | prints the project schema's path |
| `falcon export RUN... --format F --out DIR [--split-json FILE] [--option K=V]... [--overwrite]` | exports runs as `coco`, `yolo`, `kitti`, `nuscenes` (with lidarseg and panoptic), `waymo` (a documented Waymo-style JSON layout, not TFRecords), `bdd_lanes` or `mcap` (ROS 2); `falcon_export.json` in DIR says what was filled, left out and why |

Commands that need CARLA start Falcon's own server when none is running and stop it at the end (`--keep-server`
leaves it running). Ctrl+C cancels the job politely — the worker tidies the server — and a second Ctrl+C kills it.

## Exit codes

`0` ok. Otherwise the worker's code — `1` crashed, `2` failed, `3` cancelled, `4` server lost, `5` no server, `6`
refused, `7` no carla (see [the protocol](protocol.md#exit-codes)) — or `2` for a usage error or a project that does
not validate. `falcon reproduce` exits `2` when the re-run does not agree within the tolerance.

## In CI

```sh
python3 -m falcon_autonomy project validate my.falcon.json
python3 -m pytest                                   # no GPU or server needed (fake carla)
FALCON_REAL_CARLA=1 python3 -m pytest -m real_server   # on a machine with CARLA
```
