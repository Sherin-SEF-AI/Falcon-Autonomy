# The worker protocol

A worker is started as

```sh
PYTHONPATH=<falcon source> <CARLA's python> -m falcon_autonomy.workers <name> [args]
```

usually inside `systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0`. It never imports Qt. The definitive
reference is the module `falcon_autonomy/core/protocol.py`; this page summarises it.

## Worker → caller (stdout)

One JSON object per line, each with a type `t`:

| `t` | fields | meaning |
|---|---|---|
| `hello` | `worker`, `protocol` (1), `app_version`, `pid` | the worker started |
| `progress` | `f` (0–1), `text`, `eta_s` (null until known) | progress; the ETA is extrapolated from the progress so far |
| `log` | `level` (`info`, `warning`, `error`), `text` | a message for the log |
| `metric` | `name`, `value`, `unit`, `source` (`measured`, `typical`, `heuristic`) | a number with its source |
| `server` | `server_version`, `client_version`, `map`, `maps`, `synchronous_mode`, … | (probe) the server |
| `result` | `status`, `error`, `run_json`, `summary` | the last line: how it ended |

Any other line on stdout is shown as log text. stderr is the worker's log (tracebacks, libcarla's messages; on a
native crash, every Python thread's stack from `faulthandler`).

## Caller → worker (stdin)

One JSON object per line with a command `cmd`:

| `cmd` | meaning |
|---|---|
| `cancel` | tidy up and exit with `CANCELLED` (the same as SIGTERM or SIGINT) |

Live sessions (milestone 2) add `control` and `event`. The environment worker (`gym`, milestone 6) adds `reset`
(seed, options), `step` (action) and `close`; it replies on a Unix socket in the live link's framing (a JSON header
and a binary payload) — see [policies](policies.md#how-it-runs). End of input is not a cancel.

## Signals

`__main__` blocks SIGINT and SIGTERM before anything imports numpy (whose OpenBLAS starts ~23 threads at import:
a signal delivered to one of them would end the worker at once or raise a `KeyboardInterrupt` nobody sees, leaving
CARLA in synchronous mode). Only the watchdog thread takes them, with `sigtimedwait`, and sets the cancel flag; the
main thread then tidies up. If it has not finished within 15 s (a second signal: at once), the watchdog writes
`run.json` and exits with `CANCELLED`.

## The watchdog

A crashed CARLA server leaves the client hanging in its call forever. The worker reports a sign of life after every
call that returned; while it talks to the server (setup, run, tidy), no sign for longer than twice the client
time-out (or the expected time of a slow call, such as loading a map) makes the watchdog write `run.json` and exit
with `SERVER_LOST` (4).

## Exit codes

| code | status | meaning |
|---|---|---|
| 0 | ok | finished |
| 1 | crashed | an unexpected error (a bug): the traceback is in the log and `run.json` |
| 2 | failed | the run failed for a reason stated in `run.json` `error` (or a usage error) |
| 3 | cancelled | cancelled by the user (stdin `cancel`, SIGINT or SIGTERM) |
| 4 | server_lost | the CARLA server stopped answering; the watchdog ended the run |
| 5 | no_server | no CARLA server answered at the start |
| 6 | refused | refused before starting: over the render budget, not enough memory, a port not Falcon's |
| 7 | no_carla | the carla module could not be imported, or its version is not the server's |

A process killed by a signal has no exit code of its own (`-N`); Falcon then tidies the server.

## run.json

Every worker that has an output folder writes `run.json` (schema `falcon.run/1`):

```json
{"schema": "falcon.run/1", "worker": "drive", "argv": ["…/drive.json"], "status": "ok", "exit_code": 0,
 "started": "…Z", "finished": "…Z", "error": null, "notes": [],
 "provenance": {"app": "Falcon Autonomy", "app_version": "0.1.0", "created": "…Z",
                "inputs": {"config": {"path": "~/…/drive.json", "sha256": "…"}}, "seed": 7, "settings": {…},
                "host": {"python": "3.12.13", "platform": "…", "carla_server": "0.9.16", "carla_client": "0.9.16"}},
 "performance": {"steps": 600, "wall_s": 11.2, "steps_per_s": 53.6, "sim_s": 30.0, "realtime_factor": 2.68,
                 "mpx_per_step": 1.843, "mpx_per_s": 98.8, "camera_frames": 600, "encoder_waits": 0,
                 "source": "measured"},
 "summary": {…}, "name": "…", "config_sha256": "…"}
```

## The drive worker's config (`drive.json`, schema `falcon.drive_config/1`)

| field | meaning |
|---|---|
| `host`, `port`, `tm_port` | the server and the traffic manager |
| `town` | the map (loaded when another is open) |
| `weather` | every CARLA weather field (resolved from the world's preset and overrides) |
| `rig` | the rig (vehicle blueprint, sensors with poses, sizes, fields of view, label kinds) |
| `traffic` | the traffic profile: vehicle count, fleet shares by CARLA base type, behaviour |
| `vehicle_lights` | optional: `auto` (the default: position lights and low beams on every vehicle when the sun is below the horizon), `on`, `tm` (the traffic manager's own rule by weather and time of day) or `off`; run.json `lights` says what was set, how many vehicles' lights read back as on, and the map's own lights (CARLA's towns switch theirs on at night; an imported world has none). Set the same way in drives, live sessions (and again when Live's weather changes: on at nightfall, off at day), stack testing, the learning environment and hardware-in-the-loop |
| `xodr`, `speed_limits`, `keep_traffic`, `city` | an imported world: its OpenDRIVE file; traffic at each road's limit (`map`) or CARLA's own (`carla`); traffic moved away from roads that end (default true); the city plan placed around the roads ([worlds](worlds.md)) |
| `fps`, `steps`, `warmup_steps` | the fixed step (1 / fps), recorded steps, unrecorded warm-up steps |
| `max_wall_s` | stop after this much wall clock (the smoke drive); 0: run all steps |
| `seed` | spawn choices, the traffic manager, the pedestrians |
| `render_budget_mpx` | refuse the run beyond this many rendered megapixels per step |
| `out` | the run folder |

## Outputs of a drive (layout `falcon-raw`)

```
run/
  drive.json        the input
  run.json          what ran (above)
  dataset.json      sensors (intrinsics K, pose, streams), label encodings and classes, provenance
  frames.jsonl      per step: step, frame, sim_t, ego pose and speed, the files of every stream
  rgb/<sensor>/000000.jpg
  semantic/<sensor>/000000.png    uint8, the CARLA 0.9.16 semantic tag per pixel
```
