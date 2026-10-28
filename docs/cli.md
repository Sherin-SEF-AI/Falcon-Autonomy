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
| `falcon sweep PROJECT --scenario S --name N [--grid K \| --samples K] [--seed N] [--world W] [--rig R]` | a scenario sweep (a study; resumable) |
| `falcon search PROJECT --scenario S --name N [--method bayes\|cmaes\|adaptive\|random] [--budget B] [--batch K] [--threshold T]` | a rare-case search in a study (resumes, and starts from its sweeps) |
| `falcon schema` | prints the project schema's path |
| `falcon model fetch PROJECT ID` | downloads a registered model's pinned weights (URL, licence and SHA-256 printed first; refused on a hash mismatch) |
| `falcon evaluate PROJECT --model ID --run RUN... --out DIR [--camera C] [--every N] [--boot N] [--seed S] [--grid X,Y]` | scores a model on drives: COCO AP/AR (or mIoU) with drive-level bootstrap CIs, breakdowns, a condition heat map, the worst frames ([models](models.md)) |
| `falcon evaluate --compare EVAL_A EVAL_B --out DIR [--boot N]` | a paired bootstrap of every number's change B − A; regressions and improvements |
| `falcon gap --sim RUN... --real KIND:PATH --out DIR [--labels FILE] [--images N] [--features --workspace W] [--match]` | the sim-to-real gap; `falcon gap --fetch --workspace W` downloads the feature extractor ([sim-to-real](sim-to-real.md)) |
| `falcon derive worldmodel\|language RUN --out DIR [--option K=V]...` | world-model data (occupancy, BEV, scene flow, futures, actions, clips) or grounded language data ([world model](world-model.md)) |
| `falcon train --train SRC... [--val SRC] [--test SRC] [--arch A] [--init coco\|imagenet\|scratch] [--epochs N] [--batch B] [--size WxH] --workspace W` | fine-tunes a detector (a GPU job), exports ONNX and a model registration; `--weights` lists the pinned weights, `--fetch NAME` downloads one ([training](training.md)) |
| `falcon experiment mixed --id ID --synthetic SRC... --real-train SRC --real-test SRC [--shares 0,25,50,100] [--seeds K] --workspace W` | real only against real + X % synthetic at a fixed budget, paired intervals against real only (resumable) |
| `falcon experiment list\|best\|results\|export EXP [--mlflow DIR]` | runs and their metrics; the best run; a mixed experiment's table; MLflow file-store export |
| `falcon policy bc --run RUN... --out DIR` | behaviour cloning from recorded drives ([policies](policies.md)) |
| `falcon policy ppo PROJECT --out DIR [--curriculum default\|FILE \| --scenario K] [--updates N] [--steps S]` | PPO through the environment, with a scenario curriculum |
| `falcon policy rollout PROJECT --policy DIR --out DIR [--episodes N] [--scenario K]` | runs a policy for N episodes |
| `falcon policy leaderboard PROJECT --policy DIR --out DIR [--scenario K]... [--seeds-per N] [--free N]` | Leaderboard-style driving score with intervals |
| `falcon policy replay PROJECT --episodes FILE --episode N [--infraction K] --out DIR` | replays an episode's infraction with a chase camera |
| `falcon export RUN... --format F --out DIR [--split-json FILE] [--option K=V]... [--overwrite]` | exports runs as `coco`, `yolo`, `kitti`, `nuscenes` (with lidarseg and panoptic), `waymo` (a documented Waymo-style JSON layout, not TFRecords), `bdd_lanes` or `mcap` (ROS 2); `falcon_export.json` in DIR says what was filled, left out and why |
| `falcon loop PROJECT [status\|next\|run --until N\|show N\|new\|set --model ID --odd FILE]` | the closed loop: the next step's cost, run to a step, an iteration's record ([loop](loop.md)) |
| `falcon report iteration\|evaluation\|dataset\|study\|leaderboard SRC --out FILE` | a self-contained HTML report with provenance ([reports](reports.md)) |
| `falcon replay import\|match\|run\|compare …` | import a nuScenes, Waymo v2, ROS 2 bag, KITTI raw or Falcon log, check where it fits a world, replay it with changes, compare ([log replay](log-replay.md)) |
| `falcon hil device\|deploy\|session\|report\|acceptance …` | hardware in the loop: the device (address in your settings only), deploying the agent, a streaming session with latency per stage, the Jetson acceptance run ([hardware](hardware.md)) |
| `falcon stack topics PROJECT [--rig R] [--control autoware\|ackermann\|carla]` | the topics Falcon's ROS 2 bridge publishes for a rig, the control topic it takes, and the Autoware names ([stack testing](stack-testing.md)) |
| `falcon stack image` | builds the bridge image `falcon-ros2-bridge:humble` (from the official `ros:humble-ros-base`) with Docker |
| `falcon stack bridge --port P [--native] [--domain D]` | prints the bridge node's command (in the image, or `--native` on a ROS 2 machine) |
| `falcon stack run PROJECT [--scenario S --seed N] [--driver ros2\|follower\|autopilot] [--control C] [--test-stack] [--port P] [--deadline S] [--motion M] [--out DIR]` | one stack run: the ego driven through the bridge (or a baseline), closed loop, deadlines recorded |
| `falcon stack suite new NAME --out FILE [--driver D] [--scenarios a,b] [--seeds 0,1,2] [--test-stack]` | writes a suite (scenarios × seeds, pass/fail criteria) to edit |
| `falcon stack suite run PROJECT SUITE.json [--dir DIR]` | runs a suite (resumable): results.jsonl, results.json with Wilson intervals, results.csv, run.json |
| `falcon stack replay RUN [--into DIR]` | re-runs a stack run with the controls it recorded and compares the trajectories |
| `falcon stack odd new --out FILE` · `falcon stack report SUITE_DIR --odd ODD.json --out DIR [--study DIR]...` | an example ODD; the safety-case report (HTML + JSON) of a suite against it |

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
FALCON_ROS2=1 python3 -m pytest -m ros2                 # with Docker and the falcon-ros2-bridge:humble image
```
