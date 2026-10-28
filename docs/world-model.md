# World-model and language data

Two kinds of training data are derived from a recorded drive after the fact, from what it recorded. Nothing is
simulated again. `falcon derive worldmodel RUN --out DIR` and `falcon derive language RUN --out DIR` write them; options
go as `--option KEY=VALUE`. The full definitions are in the modules' docstrings (`core/worldmodel.py`,
`core/language.py`).

## World-model data (`falcon.worldmodel/1`)

All in CARLA's frames: x forward, y right, z up (left-handed), metres. "The ego frame at step i" is the ego's vehicle
frame at that step. `timeline.jsonl` lists, per step, every file of every product. `worldmodel.json` is the manifest,
with provenance over the run's run.json, frames.jsonl, lanes.json and actors.jsonl. A product whose input the run lacks
(no lidar, no actors.jsonl) is skipped, and the manifest says why.

| product | files | what |
|---|---|---|
| Occupancy | `occupancy/<lidar>/<sweep>.npz` | 0.2 m voxels over ±40 m × −2…4 m in the ego frame. The class per voxel is the labelled lidar's majority, with every actor's 3D box filled over it. The visibility mask follows Occ3D / OpenOccupancy, ray-cast with an exact voxel traversal (Amanatides & Woo): unobserved, free or occupied. A BEV view of the grid is included |
| BEV maps | `bev/<step>.npz` | ego-centred, heading up, +y to the image's right. Bit channels: drivable, centerline, lane boundary, vehicle, pedestrian, cyclist, other, ego; and per pixel the id of the actor drawn there |
| Scene flow | `flow/<lidar>/<sweep>.npy` | per lidar point, in the `.bin`'s order: its motion to the next sweep, in this sweep's sensor frame. A point moves rigidly with its actor; the map is static |
| Future trajectories | `trajectories.npz` | every agent's next 3 s at 10 Hz in the ego frame at each step, with validity masks; and the ego's own future |
| Ego actions | `ego_actions.jsonl` | the recorded control (live recordings, and drives since 0.6: the traffic manager's or the director's), and always speed, acceleration, yaw rate, curvature and lateral acceleration, derived from the poses by finite differences, and labelled so |
| Clips | `clips.json` | fixed-rate multi-camera clips (default 10 Hz, 1 s, every 0.5 s). A rate that does not divide the drive's frame rate is refused; images are never interpolated |

Camera optical flow is a label layer of its own (`flow/<camera>/<n>.npy`, see [dataset format](dataset-format.md)).

## Language data (`falcon.language/1`)

`language.jsonl` holds one record per step: captions and question–answer pairs computed from the exact simulation
state, and the facts they came from. Every caption, question and answer is grounded. Each phrase that names an
object carries the CARLA track ids it refers to, plus the character span of the phrase in the text. Phrasings vary
with a seeded generator, so the same seed gives the same text, byte for byte.

What is said, and on what basis:

- **Exact from the state:** positions, distances (box to box), directions, speeds, closing rates, counts and time to
  collision (boxes at constant velocity).
- **Lanes from lanes.json:** "your lane" is the ego's lane or its continuation along the lane graph. The others are a
  lane to the left or right (counted), oncoming, crossing, or off the roadway. Where lanes overlap, as a junction's
  turning lanes over its straight ones, an object's lane is the one whose heading matches its own. Without
  lanes.json no lane is named, and the ego's path is a corridor.
- **Why to brake (heuristic; the thresholds are in the manifest):**
  - something ahead with a time to collision under 3 s;
  - a pedestrian or cyclist in, or entering, the ego's path;
  - a red or yellow light, when recorded.

  A car closing from behind is never a reason to brake.
- **Right of way:** CARLA gives it only through traffic lights. Their states are not recorded by default, so no such
  sentence is made, and the manifest says why.
- **Scenario runs:** each timeline event is captioned at its step with the actor's CARLA id, and the run's most
  critical moment is marked.

**Paraphrasing** is optional, and runs only through a client you pass in; Falcon opens no connection of its own. A
paraphrase is accepted only if every object phrase survives verbatim (its span is re-located) and every number is
unchanged. Otherwise the template text is kept.

## Measured on a real server

On a real CARLA 0.9.16 server (2026-10-28), an 8 s Town10HD drive with a 1280 × 720 camera with every label and optical
flow, a 32-channel lidar, 40 vehicles and 20 walkers, recorded in 23 s:

- every product, nothing skipped, derived in 26 s: 80 occupancy grids, 160 BEV maps, 79 scene-flow pairs, 9600 agent
  futures (81 % of future points valid; the rest run past the drive's end) and 15 clips;
- occupancy: 0.3 % of voxels occupied, 16 % observed free, 84 % unobserved — a 32-channel lidar sees little of a
  ±40 m × 6 m volume;
- scene flow: in a mid-drive sweep, 3 % of the points move with their actors (up to 0.63 m between sweeps).

On a lead_brake scenario run, the captions follow the scenario. The lead car is "in your lane" 33 m ahead. Its braking
is captioned at the step it fires (4.85 s), it then closes at 7.1 m/s with a 3.9 s time to collision, and 3.5 s as the
ego slows. That run found, and Falcon fixed:

- a lead car on a road with another id, captioned as one lane over;
- the same car captioned as in a crossing lane inside a junction;
- kinematic actors logged in actors.jsonl as standing still.

## Limits

- The BEV map holds lanes.json's driving lanes only: no sidewalks or crossings.
- A moving actor's lidar points lie where it was during the sweep (the ego's motion is compensated, the actor's is
  not).
- Traffic-light states, and so right of way, are not recorded by default.
- Lane relations are heuristic on sharp curves.
