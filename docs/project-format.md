# Project format

A project is one JSON file, `name.falcon.json`, validated against
[`falcon_autonomy/schemas/project.schema.json`](../falcon_autonomy/schemas/project.schema.json) (JSON Schema
2020-12) on every load and save. It holds **definitions only**; runs, datasets and models live in the workspace
folder beside it (`name.falcon/`). Machine settings (CARLA's folder, ports) and secrets are never in a project.

## Top level

| field | type | meaning |
|---|---|---|
| `format` | `"falcon.project"` | identifies the file |
| `version` | integer | the format version (this build: **1**) |
| `name` | string | the project's name |
| `created`, `modified` | UTC ISO 8601 | stamps |
| `app_version` | string | the Falcon version that last saved it |
| `worlds` | [world] | towns and imported places with weather and a traffic profile |
| `traffic_profiles` | [traffic profile] | named, editable traffic |
| `rigs` | [rig] | vehicles with their sensors |
| `scenarios`, `dataset_specs`, `models`, `experiments` | [{`id`, `name`, …}] | filled by later milestones |
| `loop` | object | the closed loop's settings and iterations |
| `drive_sets` | [drive set] | optional: named sets of runs ([runs](runs.md)) |
| `trigger_rules` | [trigger rule] | optional: live trigger rules ([below](#trigger-rule), [scenarios](scenarios.md#trigger-rules)) |
| `run_tags` | {run folder: [tag]} | optional: your tags on runs, keyed by the run's folder relative to the workspace (an absolute path outside it); the runs themselves are never changed |

Ids match `^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$` and are unique within their list.

## world

| field | meaning |
|---|---|
| `id`, `name` | |
| `kind` | `town` (a CARLA map), `osm` (an OpenStreetMap import), `xodr` (an OpenDRIVE file) |
| `town` | the CARLA map name (`kind: town`), e.g. `Town10HD_Opt` |
| `osm` | the import's source and converter settings (`kind: osm`); `mesh` (the generation parameters: a world with road features at a 0.25 m vertex distance), `road_features` (the Worlds page's choices: OSM tags, what `bump` becomes, a spacing) |
| `weather` | `{"preset": "ClearNoon", …overrides}`: any of CARLA's `WeatherParameters` fields override the preset |
| `traffic_profile` | the id of a traffic profile |
| `speed_limits` | an imported world's traffic speeds: `map` (absent: the default) each lane's limit from the OpenDRIVE file, slowed ahead of lower limits and speed breakers; `carla` the traffic manager's own (30 km/h without speed signs) ([worlds](worlds.md#speed-limits-and-speed-breakers)) |
| `keep_traffic` | an imported world's traffic kept in it where roads end (absent: true): moved to a free spawn point out of the ego's view before it would drive off a dead end; `false` leaves it to CARLA (it drives off and the server removes it) ([worlds](worlds.md#roads-that-end-the-edge-of-an-imported-area)) |
| `city` | an imported world's city around the roads ([worlds](worlds.md#a-city-around-the-roads)): `enabled` (placed in every run), `plan` (`city/plan.json` in the world's folder) and `plan_sha256`, `source` (the OSM file: path, size, mtime), `seed`, `options`, `summary` (placed by kind and source, skipped and why, size closeness), `features` (counts), `timings`; absent: the roads alone |

## traffic profile

| field | meaning |
|---|---|
| `id`, `name` | |
| `handedness` | `map` (what the map's lanes say), `left` or `right` (an assertion checked against the map) |
| `vehicles`, `walkers` | how many |
| `fleet` | weights per CARLA base type (`car`, `truck`, `van`, `bus`, `motorcycle`, `bicycle`), normalised at run time |
| `behaviour` | traffic-manager settings: `distance_to_leading_m`, `speed_difference_pct`, `ignore_lights_pct`, `ignore_signs_pct`, `auto_lane_change`, `walker_crossing_factor` |

## rig and sensor

A rig: `id`, `name`, `vehicle` (a CARLA blueprint id, `vehicle.*`), `sensors`. A sensor:

| field | meaning |
|---|---|
| `id`, `name` | |
| `type` | `camera.pinhole`, `camera.fisheye`, `lidar`, `radar`, `gnss`, `imu` |
| `pose` | `x y z` metres, `roll pitch yaw` degrees, in CARLA's vehicle frame (x forward, y right, z up; left-handed), from the vehicle's origin |
| `width`, `height`, `fov` | cameras: image size in pixels, horizontal field of view in degrees |
| `labels` | cameras: label layers rendered beside the image: `semantic`, `instance`, `depth`, `boxes2d`, `boxes3d` |

This build records pinhole cameras with `semantic` labels; the other sensor types and label layers are accepted by
the schema and recorded from milestone 3 on.

## Versions and migrations

`version` is checked on load. A file from an older format is migrated in memory (the project is then marked
modified, and the log says so) and written in the current format on the next save; `falcon project migrate PATH`
does it in place. A file from a newer format than the build knows is refused with a message to update Falcon.

| from → to | what changes |
|---|---|
| 0 → 1 | rigs' `cameras` become typed `sensors`; traffic profiles and empty sections are added |

## Edits

Every change made in the app is an edit (`set` a value, `insert` into or `remove` from a list) at a path into the
document, e.g. `["worlds", 0, "weather"]`; each edit knows its inverse, which is how undo and redo work.

## drive set

| field | meaning |
|---|---|
| `id`, `name` | |
| `kind` | `list` (the runs named in `runs`) or `query` (the runs matching `query` whenever the set is used) |
| `runs` | `kind: list`: run folders relative to the workspace (absolute outside it); one no longer on disk drops out |
| `query` | `kind: query`: a search in the run index's language ([runs](runs.md)), e.g. `weather:rain tod:night status:ok` |
| `created` | UTC ISO 8601 |

## trigger rule

| field | meaning |
|---|---|
| `id` | unique among the rules; `fired.<id>` in another rule's condition counts its firings |
| `name`, `about` | optional text |
| `rule` | `when <condition> -> <action>[; <action>]` ([trigger rules](scenarios.md#trigger-rules)); checked by `core.triggers`, not by the schema, so a rule being typed can be saved — a run refuses an enabled rule that does not parse |
| `repeat` | `once` (default), `every` (each time the condition becomes true) or `cooldown` (while it holds, once per `cooldown_s`) |
| `cooldown_s` | seconds ≥ 0 (> 0 for `cooldown`) |
| `enabled` | default true |
| `applies` | some of `drives`, `scenarios`, `live` (default all three) |

Added after 2.3.0 without a format version change: an older build refuses a project that has `trigger_rules` (its schema
does not know the field); removing the field makes the file readable there again.
