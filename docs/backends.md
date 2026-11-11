# CARLA backends

Falcon runs on one of two simulators: **CARLA 0.9.16** (Unreal Engine 4.26), the default and what every figure in this
documentation was measured on, or **CARLA 0.10** (CARLA's Unreal Engine 5 line, on UE 5.5). The second is an adapter
tested against a fake; it has not yet been run against a real 0.10 server. Settings ▸ **Simulator**; `falcon backend
list|check|probe`. Code: `falcon_autonomy/sim/backends.py` (the table), `sim/server.py` (the server per backend),
`sim/service.py` (`CarlaService.backend`, `use_backend`), `workers/compat.py` (what a worker adapts); their docstrings
are the reference.

There is no "CARLA UE5.8": the UE5 line (0.10.x) builds on UE 5.5. It ships two maps (Town10HD_Opt and Mine_01) and no
native ROS 2. Scenario code is the same on both; a world the backend lacks is refused, with the reason.

## Which backend a run uses

1. The project's simulator (`simulator.backend` in the project file; written only when it is not 0.9.16), else
2. the setting **Default backend** (`default_backend`).

A drive config made for a project that names its backend carries it (`backend`, absent for 0.9.16), and it is part of
the drive's cache key (`core.datasets.drive_key`): a 0.10 drive never stands in for a 0.9.16 one, and every 0.9.16 key
is what it was before. A run config made for one backend and started on the other is refused before the world loads.

Each backend has its own CARLA folder and Python: 0.9.16's are Settings ▸ CARLA (`carla_root`, `client_python`, as
before); the others' are Settings ▸ Simulator (`carla_roots`, `client_pythons`, by backend). Empty means the first folder
of that backend found in the usual places (`~/carla`, `~/CARLA*`, `~/Carla*` for 0.10, `/opt/carla*`, …) and the venv
beside it. A worker is started with `FALCON_BACKEND` naming its backend and refuses a client module of the other one
(exit 7, no carla).

## What each can do

| | CARLA 0.9.16 | CARLA 0.10 (UE5) |
|---|---|---|
| Server | `CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping CarlaUE4` | `CarlaUnreal/Binaries/Linux/CarlaUnreal-Linux-Shipping CarlaUnreal` |
| Client | `pip install carla==0.9.16` | the wheel in the release's `PythonAPI/carla/dist` (0.10.x) |
| Maps | whatever the server lists | Town10HD_Opt, Mine_01 |
| Native ROS 2 (`--ros2`) | yes (measured) | no |
| OSM import (`carla.Osm2Odr`) | yes | no |
| Traffic manager OSM mode | yes | no |
| Lane offset (lateral filtering) | yes | assumed yes |
| Imported (OpenDRIVE) worlds | yes | refused (not checked) |
| Falcon's measured figures | all | none |

The UE5 column is the adapter's assumption until `falcon backend probe carla-ue5` reads it from a server: the probe
records the maps, blueprint ids, `CityObjectLabel` names and values, weather fields, camera and lidar attributes, and
the client module's API (`Osm2Odr`, `TrafficManager.set_osm_mode` and `vehicle_lane_offset`, `Sensor.enable_for_ros`)
in `$XDG_CONFIG_HOME/FalconAutonomy/backends/carla-ue5.json`; those maps and API capabilities then replace the declared
ones. Imported worlds stay refused: `generate_opendrive_world` exists in both client modules, but whether a 0.10 server
builds a world from it was not checked.

## What the adapter does on 0.10

On 0.9.16 nothing changes: the adapter is the identity, the server command is exactly what Falcon always started, and
run.json only gains `provenance.backend` (`{"id": "carla-0.9.16", "carla_client": "0.9.16"}`). On 0.10 a worker:

- **Blueprints**: a 0.9.16 id the server lacks is looked up in `resources/carla/blueprints-carla-ue5.json` (renamed
  ids, e.g. `vehicle.lincoln.mkz_2020` → `vehicle.lincoln.mkz`), then replaced by the first blueprint (by id) of its
  0.9.16 base type. The table was not read from a 0.10 server. Every substitution is recorded (`compat.substitutions`)
  and noted once.
- **Semantic tags**: `carla.CityObjectLabel` is read at run time and the tags the instance camera and the semantic lidar
  write are mapped to Falcon's classes by label name (`core.groundtruth.tags_for`, `tag_lut`; case, underscores and a
  plural *s* ignored). A label with no Falcon class is labelled unlabelled and named in the notes. Its values are never
  assumed: on the fake, whose values are shuffled, the labels come out pixel for pixel as on 0.9.16.
- **Worlds**: a map the server does not list, an imported world and an OSM import are refused (exit 6) before anything
  loads; lateral filtering turns off with the reason when the traffic manager has no lane offset.
- **Weather**: a field the backend's `WeatherParameters` lacks is noted when a run asks for it (non-zero).
- **Cameras**: a lens or post-process attribute a fisheye view asks for (`core.lens.VIEW_ATTRIBUTES`) that the blueprint
  lacks is left at the engine's own and noted; the views may then differ at their seams.
- **Tone mapping**: the vignette correction is 0.9.16's measured camera response (`sim/tonemap.py`); on 0.10 it is off
  and noted.
- **Traffic keeper**: its look-ahead and fall limits were measured with 0.9.16's traffic manager; on 0.10 that is
  noted.

run.json `compat` lists the backend, the substitutions and the notes.

## `falcon backend`

```
falcon backend list [--json]                  each backend: folder, Python, maps, capabilities, last probe
falcon backend check [ID] [--root DIR] [--python PY] [--json]
                                              the folder holds the backend's server; its Python imports its client
                                              (a 0.9.16 wheel is not a 0.10 client); 0 ok, 2 not
falcon backend probe [ID] [--wait S] [--keep-server] [--json]
                                              start Falcon's own server of that backend (when none runs) and record
                                              what it has; refused when the server answers as the other backend
```

`check` and `probe` default to the setting's backend.

**On a cluster** ([cluster.md](cluster.md#scheduling)) each worker daemon's hello names the backends its machine has
— a backend whose folder holds its server and whose client Python is there — with their capabilities and maps as
above (declared, or a probe's), and the scheduler sends a job only to a node that has the job's backend and the map
and capabilities its run needs: a project on CARLA 0.10 with no 0.10 node is refused ("no node has carla-ue5 (ws1/0
has carla-0.9.16)…"). Tested with the UE5 fake behind one local daemon and the 0.9.16 fake behind another.

## Tests and limits

`tests/fake_carla_ue5` is the 0.9.16 fake run in its own namespace and changed where the adapter has to cope: version
0.10.0, renamed and missing blueprints, shuffled `CityObjectLabel` values, no `Osm2Odr` / `set_osm_mode` /
`enable_for_ros`, no `dust_storm`, cameras without `lens_*` attributes, two maps; `tests/fake_carla_ue5/root` holds a
stand-in folder with `CarlaUnreal-Linux-Shipping` (a stub) for detection. These differences exercise the adapter; none
is a statement of what a real 0.10 server does. `tests/test_backends.py` runs a smoke drive, a scenario sweep and a
benchmark route on Town10HD_Opt on both fakes through the unchanged service, study and benchmark paths.

Limits: nothing on 0.10 is measured — timings, video memory, the tone curve and vignette, the traffic manager's
behaviour, the blueprint names and the server's switches (`-RenderOffScreen`, `-quality-level=Low`, `-nullrhi` are
0.9.16's, passed unchanged). Tests marked `ue5` (`FALCON_REAL_UE5=1`) check and probe a real installation; they have
not been run.
