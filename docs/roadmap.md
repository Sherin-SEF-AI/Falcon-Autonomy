# Roadmap

Each milestone ends with its tests passing, its docs and a demo. Pages appear in the app only when they work.

| # | milestone | contents | status |
|---|---|---|---|
| 1 | Foundation | app shell, project format, job system, CARLA server manager, workers, fake carla, first-run wizard, a basic drive with RGB and semantic labels, CLI | **done (0.1.0)** |
| 2 | Worlds and Live | towns and weather; OpenStreetMap import (central Bengaluru first) with the road-graph preview; the Indian traffic profile and verified traffic handedness; the live view, manual driving, live events, recording | next |
| 3 | Rigs and data engine | the rig editor (pinhole, fisheye, lidar, radar, GNSS, IMU), every label type, exports checked with the official devkits, the dataset browser, coverage specs | |
| 4 | Scenarios and rare cases | the studio, built-in scenarios (including the Indian ones), criticality measures, sweeps, search | |
| 5 | Models and the gap | model plug-ins, the test harness, sweeps, sim-to-real gap, BEV / occupancy / world-model data, language data | |
| 6 | Training lab | perception fine-tuning, the Gymnasium environment, imitation and reinforcement learning, Leaderboard-style evaluation | |
| 7 | Stack, replay, hardware | ROS 2 stack testing, log replay with changes, Jetson in the loop | |
| 8 | The loop | the home-page orchestrator, reports, packaging, release 1.0 | |
