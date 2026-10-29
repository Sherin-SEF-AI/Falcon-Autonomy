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
| `falcon search PROJECT --scenario S --name N [--method bayes\|bayes-pof\|cmaes\|adaptive\|random] [--budget B] [--batch K] [--threshold T] [--objective criticality\|perception\|both --model ID] [--weather FIELD[:LO:HI]] [--metric M] [--fps F] [--search-seed N]` | a rare-case search in a study (resumes, and starts from its sweeps) |
| `falcon rules check RUN... [--set india\|generic\|FILE] [--out DIR]` · `falcon rules list\|show\|signals` | checks recorded runs (or an episodes.jsonl) against traffic rules written in temporal logic: robustness, violation intervals, counts per km ([traffic rules](traffic-rules.md)) |
| `falcon experiment active --id ID --pool SRC... --test SRC [--strategies entropy,margin,kcenter,hybrid] [--rounds R] [--budget B] [--seeds K] --workspace W` | active learning: which frames to add, against random selection with paired intervals ([active learning](active-learning.md)) |
| `falcon recon train\|eval\|render\|rerender\|info …` | neural reconstruction of a recorded drive (3D Gaussian Splatting in pure PyTorch): train, held-out and novel-view quality, render a pose, re-render the drive along a shifted path ([reconstruction](reconstruction.md)) |
| `falcon worldmodel train\|eval\|imagine\|collect\|e2e\|e2e-offline\|show …` | a learned world model (RSSM) on drives and episodes, scored against copy-last and constant-velocity baselines per horizon; policies learnt in its imagination; end-to-end camera policies ([world-model training](world-model-training.md)) |
| `falcon adversary train\|eval\|replay …` | learned adversarial traffic under realism constraints against a fixed ego; failure rates against random and scripted baselines; replays ([adversarial traffic](adversarial-traffic.md)) |
| `falcon vlm list\|fetch\|verify\|eval\|finetune …` | vision-language models scored against the simulator's truth (accuracy, grounding, hallucination, faithfulness); approved downloads only ([vision-language](vlm.md)) |
| `falcon deploy check\|calibrate\|build\|profile\|accuracy\|gate\|report\|all …` | TensorRT engines (FP32/FP16/INT8, GPU or DLA) built and profiled on the Jetson AGX Orin — latency, power and energy from its rails, accuracy against the host's ONNX model with paired intervals — and a release gate; a labelled simulated device (onnxruntime on this host) for the pipeline only ([deploy](deploy.md)) |
| `falcon why case PROJECT --run RUN --out DIR [--failure F] [--model ID] [--actor A \| --track N] [--step S]` | why a scenario run failed: counterfactual re-runs, one factor at a time ([root cause](root-cause.md)) |
| `falcon why evaluation EVAL --out DIR` | the correlates of an evaluation's misses, with intervals |
| `falcon why episode PROJECT --episodes FILE --episode N --policy P --out DIR` | a Leaderboard infraction by the policy's inputs |
| `falcon assets list` · `show CLASS` · `inventory [--out FILE]` · `verify --out DIR [--stages S,…]` | Indian road assets: the stand-ins and proxies with their sources; measure what Falcon's own server can spawn; the road-feature and asset checks ([Indian road assets](indian-assets.md)) |
| `falcon schema` | prints the project schema's path |
| `falcon model fetch PROJECT ID` | downloads a registered model's pinned weights (URL, licence and SHA-256 printed first; refused on a hash mismatch) |
| `falcon evaluate PROJECT --model ID --run RUN... --out DIR [--camera C] [--every N] [--boot N] [--seed S] [--grid X,Y]` | scores a model on drives: COCO AP/AR (or mIoU) with drive-level bootstrap CIs, breakdowns, a condition heat map, the worst frames ([models](models.md)) |
| `falcon evaluate --compare EVAL_A EVAL_B --out DIR [--boot N]` | a paired bootstrap of every number's change B − A; regressions and improvements |
| `falcon gap --sim RUN... --real KIND:PATH --out DIR [--labels FILE] [--images N] [--features --workspace W] [--match]` | the sim-to-real gap; `falcon gap --fetch --workspace W` downloads the feature extractor ([sim-to-real](sim-to-real.md)) |
| `falcon odd calibrate --in RUN... [--out-of NAME=RUN[,RUN]...] [--far A] [--every N] [--features WS] --out FILE` | an ODD data-monitor reference from in-ODD drives, its threshold for the false-alarm rate A, and the measured false-alarm and detection rates ([ODD monitoring](odd-monitoring.md)) |
| `falcon odd monitor RUN... --odd FILE [--reference REF] [--every N] [--camera ID] [--no-write]` | the ODD monitor over recorded runs: `odd-monitor.json` in each, the time outside and why |
| `falcon derive worldmodel\|language RUN --out DIR [--option K=V]...` | world-model data (occupancy, BEV, scene flow, futures, actions, clips) or grounded language data ([world model](world-model.md)) |
| `falcon train --train SRC... [--val SRC] [--test SRC] [--arch A] [--init coco\|imagenet\|scratch] [--epochs N] [--batch B] [--size WxH] --workspace W` | fine-tunes a detector (a GPU job), exports ONNX and a model registration; `--weights` lists the pinned weights, `--fetch NAME` downloads one ([training](training.md)) |
| `falcon experiment mixed --id ID --synthetic SRC... --real-train SRC --real-test SRC [--shares 0,25,50,100] [--seeds K] --workspace W` | real only against real + X % synthetic at a fixed budget, paired intervals against real only (resumable) |
| `falcon experiment list\|best\|results\|export EXP [--mlflow DIR]` | runs and their metrics; the best run; a mixed experiment's table; MLflow file-store export |
| `falcon translate train --sim RUN... --real KIND:PATH --workspace W [--id ID] [--iterations N]` · `apply --model DIR --run RUN... --workspace W [--threshold T] [--on-drift raw\|flag]` · `check RUN... [--real KIND:PATH --features] [--out DIR]` · `experiment --id ID --synthetic SRC... --translated SRC... --real-train SRC --real-test SRC` · `results EXP` | learned sim-to-real image translation (CUT) with every label kept: train a translator (a GPU job), translate drives into new runs, measure each frame's drift and the gap before and after, translated against raw synthetic on a real test set ([translation](translation.md)) |
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
| `falcon ci run CONFIG [--out DIR] [--mode falcon\|fake] [--fake-carla DIR] [--baseline DIR \| --no-baseline] [--force]` | CI mode: a stack's suite (or a policy's routes, or a model's evaluation) headless, judged by gates; JUnit XML, a Markdown summary, an HTML report, results JSON; exit 0 passed, 1 a gate failed, 2 error, 3 cancelled ([CI mode](ci.md)) |
| `falcon ci validate\|schema\|summary\|lint …` · `falcon ci init [TARGET] [--kind K] [--github]` · `falcon ci post RESULTS [--as comment\|check\|status] [--post]` | check a CI config, print its JSON Schema or a summary, check a workflow; write a config, a suite and a GitHub Actions workflow into a repository; post the summary to GitHub (a dry run unless `--post`) |

Commands that need CARLA start Falcon's own server when none is running and stop it at the end (`--keep-server`
leaves it running). Ctrl+C cancels the job politely — the worker tidies the server — and a second Ctrl+C kills it.

## Exit codes

`0` ok. Otherwise the worker's code — `1` crashed, `2` failed, `3` cancelled, `4` server lost, `5` no server, `6`
refused, `7` no carla (see [the protocol](protocol.md#exit-codes)) — or `2` for a usage error or a project that does
not validate. `falcon reproduce` exits `2` when the re-run does not agree within the tolerance. `falcon ci run` follows
a test runner's convention instead: `0` every gate passed, `1` a gate failed, `2` an error (no verdict), `3` cancelled
([CI mode](ci.md#exit-codes)).

## In CI

```sh
python3 -m falcon_autonomy project validate my.falcon.json
python3 -m pytest                                   # no GPU or server needed (fake carla)
FALCON_REAL_CARLA=1 python3 -m pytest -m real_server   # on a machine with CARLA
FALCON_ROS2=1 python3 -m pytest -m ros2                 # with Docker and the falcon-ros2-bridge:humble image
```
