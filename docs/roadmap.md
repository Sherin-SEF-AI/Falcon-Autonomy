# Roadmap

Each milestone ends with its tests passing, its docs and a demo. Pages appear in the app only when they work.

| # | milestone | contents | status |
|---|---|---|---|
| 1 | Foundation | app shell, project format, job system, CARLA server manager, workers, fake carla, first-run wizard, a basic drive with RGB and semantic labels, CLI | **done (0.1.0)** |
| 2 | Worlds and Live | towns and weather; OpenStreetMap import (central Bengaluru first) with the road-graph preview; the Indian traffic profile and verified traffic handedness; the live view, manual driving, live events, recording | **done (0.2.0)** |
| 3 | Rigs and data engine | map-based ground relabelling for imported worlds; the rig editor (pinhole, fisheye, lidar, radar, GNSS, IMU), every label type, exports checked with the official devkits, the dataset browser, coverage specs | **done (0.3.0)** |
| 4 | Scenarios and rare cases | the studio, built-in scenarios (including the Indian ones), criticality measures, sweeps, search | **done (0.4.0)** |
| 5 | Models and the gap | optical flow and scene flow; model plug-ins, the test harness, sweeps, sim-to-real gap, BEV / occupancy / world-model data, language data | **done (0.5.0)** |
| 6 | Training lab | perception fine-tuning (with mixed sim/real training experiments), the Gymnasium environment, imitation and reinforcement learning, Leaderboard-style evaluation | **done (0.6.0)** |
| 7 | Stack, replay, hardware | ROS 2 stack testing, log replay with changes, Jetson in the loop | **done (1.0.0)** |
| 8 | The loop | the home-page orchestrator, reports, packaging, release 1.0 | **done (1.0.0)** |

## After 1.0

| # | milestone | contents | status |
|---|---|---|---|
| 9 | Sharper loop | model-in-the-loop rare-case search, automatic root-cause attribution, active learning, formal traffic-rule checking (temporal logic, Indian rules) | **done (1.1.0)** |
| 10 | Realism | weather and sensor physics CARLA lacks (lidar and radar in rain and fog, lens droplets, glare), Indian road assets within the packaged CARLA, runtime ODD monitoring | **done (1.2.0)** |
| 11 | Real to sim | learned sim-to-real image translation (labels kept), neural reconstruction of real logs | **done (1.3.0)** |
| 12 | Learned worlds | world-model / end-to-end training, adversarial multi-agent traffic, vision-language driving explanations scored against ground truth | **done (1.4.0)** |
| 13 | Deployment | TensorRT export and profiling on the Jetson AGX Orin, CI mode for stacks on GitHub, release 2.0 | next |
