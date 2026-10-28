# Architecture

Falcon is **two kinds of process** that never share an interpreter:

| Process | Python | Imports | Never imports |
|---|---|---|---|
| the GUI / the CLI | the system's (3.12+; 3.14 on the reference machine) | PyQt6, numpy, OpenCV, jsonschema | `carla` |
| a worker | CARLA's venv (3.12 with the carla 0.9.16 wheel) | `carla`, numpy, OpenCV | Qt |

They talk over the [local protocol](protocol.md): JSON lines on stdin/stdout (and, for live video, a local socket).

## Packages

- `falcon_autonomy.core` — process-neutral (standard library, numpy where noted): the project format and its edits,
  provenance, the catalogue (SQLite), jobs (queue, locks, memory guard), budgets and cost, weather, labels, drive
  configs, reproduction checks. Imported by the GUI, the CLI and the workers.
- `falcon_autonomy.sim` — the GUI side of CARLA: finding it, Falcon's own server, worker commands, and
  `CarlaService`, which turns "run this on CARLA" into a queued job for the GUI and the CLI alike.
- `falcon_autonomy.workers` — run in CARLA's Python: `probe`, `tidy`, `drive` (more per milestone). Entry point
  `python -m falcon_autonomy.workers NAME ARGS`.
- `falcon_autonomy.ui` — the PyQt6 application: the main window, pages, the wizard, the palette.
- `falcon_autonomy.cli` — the command line ([reference](cli.md)).

## A job's life

1. A page (or a CLI command) builds a config with `core` (e.g. `core.drive.drive_config`) and hands it to
   `CarlaService`, which writes it into a new run folder and submits a `JobSpec` (the worker command, what it needs).
2. `JobQueue.poll()` starts it when what it needs is free: the machine-wide CARLA lock, the GPU lock, memory. Before
   a CARLA job starts, the service makes sure Falcon's server is up (starting it under its cap).
3. The launcher (QProcess in the GUI, subprocess in the CLI) feeds the worker's protocol lines back to the queue;
   the Jobs dock and the CLI show progress, ETA and logs.
4. The worker writes `run.json` and exits with a [documented code](cli.md#exit-codes). The service indexes the run
   in the catalogue, updates the measured throughput, and tidies the server if the worker died without doing so.

## Storage

- The **project file** (`*.falcon.json`, [format](project-format.md)) holds definitions only.
- The **workspace** (`*.falcon/` beside it): `runs/<stamp-name>/` (each with `drive.json`, `run.json`, the data),
  `logs/`, `catalog.sqlite`.
- The **catalogue** indexes runs (and, as milestones land, datasets, models, evaluations, experiments); it can be
  rebuilt from the files (`falcon runs PROJECT --rebuild`), which are the truth.
- **User settings and secrets** live in `$XDG_CONFIG_HOME/FalconAutonomy/`; secrets in a separate file readable
  only by the user, never shown and never copied into a project.
