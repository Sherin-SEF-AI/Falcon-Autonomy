# Architecture

Falcon is **two kinds of process** that never share an interpreter:

| Process | Python | Imports | Never imports |
|---|---|---|---|
| the GUI / the CLI | the system's (3.12+; 3.14 on the reference machine) | PyQt6, numpy, OpenCV, jsonschema | `carla` |
| a worker | CARLA's venv (3.12 with the carla 0.9.16 wheel) | `carla`, numpy, OpenCV | Qt |

They talk over the [local protocol](protocol.md): JSON lines on stdin/stdout (and, for live video, a local socket).

**A worker runs in its backend's Python.** Falcon has two CARLA backends ([backends](backends.md)): 0.9.16, the
default, and CARLA's UE5 line (0.10.x). `sim/backends.py` is the table — each backend's server binary inside its CARLA
folder, its client wheel, maps and capabilities — and is standard library only, so the GUI, the CLI and the workers
all read it. The service starts a worker with the Python of the run's backend (Settings ▸ CARLA for 0.9.16, Settings ▸
Simulator for the others) and names the backend in `FALCON_BACKEND` (`sim.server.worker_command`). Inside the worker,
`workers/compat.py` attaches a `Compat` when `carla` is imported: on 0.9.16 every method is the identity; on 0.10 it
refuses a client module of the other backend (exit 7), maps blueprints and semantic tags, refuses what the backend
cannot do, and records each adaptation in run.json (`compat`).

Two more kinds of process exist since 4.0, each optional:

- **A worker daemon** (`falcon worker-daemon serve --slot N`, [cluster](cluster.md)) on a compute node: the system's
  Python, listening on 127.0.0.1:2350+N only, running allow-listed workers from bundles through its own `CarlaService`
  in a scratch workspace; `oneshot` runs one bundle for a Slurm task, a Kubernetes pod or a cloud burst.
- **The MCP server** (`falcon --mcp`, [MCP](mcp.md)), started by Claude Code or another MCP client on stdin and stdout.
  With a Falcon window open it sends each tool call to the window's **bridge** (`ui/mcp_bridge.py`: 127.0.0.1 only, a
  random token in `<config>/mcp/<pid>.json`, mode 0600), which runs it on the window's project on the GUI thread as
  one undo step; without a window it runs the tools itself on a project file (`mcp/session.py` `FileSession`), saving
  each act call and backing the file up first.

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
- `falcon_autonomy.cluster` (4.0) — the distributed worker mesh and cloud bursts, standard library only (paramiko for
  SSH, imported when a node needs it): `protocol` (the daemon's framed messages and compressed artifacts), `bundle`
  (what a remote job carries: its config and the content of its inputs, never a path; anything dashcam-derived
  refused), `nodes` (the private `cluster` setting), `daemon`, `transport` (key-only SSH tunnels), `scheduler` (slots,
  matching each job to a node that has its CARLA backend, heartbeats, re-queue, pools and quotas), `launcher`
  (`RemoteLauncher`, the job queue's door to the scheduler), `slurm` and `k8s` (batch nodes), `cloud` with `runpod`
  and `lambda_` (estimates, approved bursts, the ledger, the watchdog), `gc` (the caches nothing refers to: the
  content store, nodes' scratch, Slurm array folders), `cli` and `cloudcli` ([cluster](cluster.md),
  [cloud](cloud.md)).
- `falcon_autonomy.mcp` (4.0) — Falcon as an MCP server: `server` (the protocol, ported from Vantage), `service` (the
  assistant's tools and their refusals), `session` (the window or a file), `catalog` (the `falcon://` resources)
  ([MCP](mcp.md)).
- `falcon_autonomy.runtime` — the bird's-eye reference stitcher `bev_stitch.py`, copied unchanged from Vantage and
  licensed **Apache-2.0** (`LICENSE-APACHE-2.0` beside it; the rest of Falcon is GPL-3.0-or-later). It needs only numpy
  and OpenCV and is what `falcon bev export` puts beside the tables for the vehicle; Falcon's own stitcher is
  `core/bevstitch.py` (lens-model remap tables over the ground, seams, feathered weights, a batched `grid_sample` on
  CUDA or OpenCV's remap; [surround view](bev.md)).
- `falcon_autonomy.core.recon` — neural reconstruction ([reconstruction](reconstruction.md)); since 4.0 also
  `actors.py` (tracked road users as Gaussians in their box frames under learned pose corrections, wheels,
  pedestrians' deformation), `lidar.py` (the neural lidar: beams rendered with ray drop and intensity),
  `lens_raster.py` (fisheye rendering by cube faces or through the lens model), `closedloop.py` (`NeuralSim`: a policy
  driving through a reconstruction, writing ordinary run folders; road users from an `ActorController`) and
  `dashcam_tracks.py` (a near-miss ingest's tracks carried into a dashcam reconstruction's world frame by frame,
  A_recon = C_recon · E_nearmiss⁻¹ · A_nearmiss), and `levelk_actors.py` (an `ActorController` whose road users are level-k agents
  deciding against the policy's ego). `core/levelk_nearmiss.py` calibrates level-k on a near-miss ingest's tracks.

## A job's life

1. A page (or a CLI command) builds a config with `core` (e.g. `core.drive.drive_config`) and hands it to
   `CarlaService`, which writes it into a new run folder and submits a `JobSpec` (the worker command, what it needs).
2. `JobQueue.poll()` starts it when what it needs is free: the machine-wide CARLA lock, the GPU lock, memory. Before
   a CARLA job starts, the service makes sure Falcon's server is up (starting it under its cap). Before a GPU job
   (training, reconstruction, translation, evaluation) it stops Falcon's own server, which is idle then but holds
   its video memory: 5.9 of 16 GB on an RTX 5080 for CARLA 0.9.16 on its default map, measured 2026-10-30. It
   stops in 0.7 s; the next CARLA job starts it again (3 s, plus its map). A server the user started, or one
   another Falcon process is using, is left running. Settings ▸ *GPU jobs* turns this off (`free_gpu_for_jobs`).
3. The launcher (QProcess in the GUI, subprocess in the CLI) feeds the worker's protocol lines back to the queue;
   the Jobs dock and the CLI show progress, ETA and logs.
4. The worker writes `run.json` and exits with a [documented code](cli.md#exit-codes). The service indexes the run
   in the catalogue, updates the measured throughput, and tidies the server if the worker died without doing so.
   With `lineage_auto_index` on (off by default: the content store hard-links a run's frames, which makes them
   read-only), it also queues the index of the run's frames into the workspace's content store.

## A remote job's life

A study run with `--on cluster[:POOL]` (or a pool chosen in a panel's *Runs on*) stays on this machine and decides
what to drive next; only its drives go elsewhere ([cluster](cluster.md)).

1. **Bundle.** The service builds the drive's `JobSpec` as for a local run, with its placement. `cluster.bundle`
   turns it into a bundle: the config without what only makes sense here (its output folder, the server's address
   and ports) and the content of every file the config names, each with its SHA-256. A worker outside the allow-list,
   an interactive job, a config naming a folder, anything derived from the dashcam footage, or a bundle over 5 MiB
   compressed is refused before anything is sent.
2. **Node.** `JobQueue.poll()` asks `cluster.launcher.RemoteLauncher` whether the job can start; the scheduler picks
   a free slot of a node in the job's pool that has its CARLA backend, map and capabilities (or waits, or refuses
   saying what each node has), and submits the bundle — over an SSH tunnel to the daemon's 127.0.0.1 port with the
   slot's token, as an `sbatch` array task, or as a Kubernetes Job. The node checks every file's hash, builds the
   worker's command itself, runs it under its own `CarlaService`, and streams each protocol line back (`msg`) with a
   heartbeat every 15 s. A slot silent for 60 s is lost and a resumable job is queued again, at most twice.
3. **Artifact.** When the worker ends, the node packs its run folder (zstd when both ends have Python 3.14's
   `compression.zstd`, else zlib) and sends it back (`done`).
4. **Unpack.** The folder is unpacked into the job's own run folder here — refusing absolute names, `..`, links and
   devices — before the queue hears that the job ended; the node's paths in its text files become this machine's, and
   run.json gains a `node` block (node, kind, pool, slot, dispatch id, host name, Falcon version, the node's own
   settings and config hashes).
5. **Catalogue.** From there the job ends as a local one does: the service indexes the run in the catalogue and the
   study reads its results; apart from the `node` block (and timings) the run is the one a local run writes.

A cloud burst is the same path one level up: a whole study is bundled (`study` bundle), run by `oneshot` on the rented
machine, and its summary — the study's files and its ten most critical runs' records — comes back into
`studies/<name>/` with `cloud.json` ([cloud](cloud.md)).

## Storage

- The **project file** (`*.falcon.json`, [format](project-format.md)) holds definitions only.
- The **workspace** (`*.falcon/` beside it): `runs/<stamp-name>/` (each with `drive.json`, `run.json`, the data),
  `logs/`, `catalog.sqlite`.
- The **catalogue** indexes runs (and, as milestones land, datasets, models, evaluations, experiments); it can be
  rebuilt from the files (`falcon runs PROJECT --rebuild`), which are the truth. In a [team workspace](team.md) on a
  shared folder it is a per-user copy in the cache folder (SQLite is not safe on NFS or SMB), rebuilt on open; the
  shared files (`team/`: comments, triage events, run owners) are append-only, one file each.
- The **experiment registry** is the workspace's `experiments/<id>/` folders (each with its definition and hash, plan
  and pins, tracked runs and results) and the catalogue's experiment rows ([experiments](experiments.md#the-registry));
  outputs for papers go to `paper/`.
- A scenario **written from a description** keeps its record (every attempt's validation errors, what was sent) in
  `scenario-text/<id>.json`; a run checked against a description, its report in `scenario-check.json`
  ([a scenario from a description](scenario-text.md)).
- The **content store** (4.0, [lineage](lineage.md)): `cas/objects/<first two hex>/<hex>` in the workspace, each
  object read-only (0444) under the address of its bytes (`s256:` SHA-256, or `b3:` BLAKE3 with the `lineage` extra),
  put by reflink, else hard link, else copy; `cas/tmp/` for puts in progress, `cas/lock` (shared for puts, exclusive
  for gc). A run's index is `RUN/frames.parquet`; a dataset version's samples `datasets/<spec>/splits/vNNN.parquet`
  with its Merkle roots in the version's JSON; a training's per-sample addresses in its `data.json`, and its root in
  the exported ONNX file's metadata. Those three are the store's only roots: `falcon lineage gc` and `falcon cluster gc`
  remove objects none of them names.
- The **safety case** (4.0): `safety/gsn.json` (the argument and its bound evidence), `safety/gsn-events.jsonl`
  (append-only), `safety/formulas.json` (saved STL requirements) ([safety case](gsn.md)); falsification studies are
  ordinary studies, `studies/falsify-<scenario>-<key>/` with `falsify.json` ([falsification](falsification.md)).
- **Introspection** outputs go to `introspect/` (`attention-…`, `intervene-…`, `audit-…`).
- **Near-miss mining** writes only inside the folder of clips it reads, under `falcon-ingest/` (private, like
  everything made from the footage); a scenario from an event goes to the project's `incidents/<name>/`
  ([near-misses](nearmiss.md)).
- **User settings and secrets** live in `$XDG_CONFIG_HOME/FalconAutonomy/`; secrets in a separate file readable
  only by the user, never shown and never copied into a project. Since 4.0 the same folder holds `backends/<id>.json`
  (a backend's probe), `profile-saved.json` (the values a profile replaced), `cluster/` (Falcon's own `known_hosts`,
  the pools' daily counts, a daemon's token), `cloud/` (the append-only, fsynced burst ledger and each burst's own key)
  and `mcp/<pid>.json` (an open window's MCP bridge: its port and token).
