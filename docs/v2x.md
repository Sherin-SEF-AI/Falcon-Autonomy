# Cooperative V2X: roadside units

A **roadside unit** (RSU) is a pole standing in a world — a smart-pole camera, an intersection lidar, or both — that is
not on the ego. Falcon records its sensors in every drive of that world with the same labels as the ego's, and makes
from what they actually see a **Collective Perception Message** (CPM) in the shape of ETSI TS 103 324 V2.1.1, UPER-
encoded, every message period. The stack bridge publishes the messages over ROS 2; runs and MCAP exports keep them.
How much the ego gains from them is measured, below, at a busy junction of central Bengaluru.

What this is not: the CPMs are built from the simulator's ground truth filtered by what the RSU's sensors reached — a
perfect detector and tracker behind real sensor geometry and occlusion, with optional Gaussian noise — not the output
of a perception stack; nothing models the radio channel, latency or loss; and the encoding has been checked against
two independent decoders, which is **not a conformance test** of the standard (see *The encoding*).

## Defining roadside units

The project's `infrastructure.rsus` ([project format](project-format.md#roadside-unit)): per pole its world, the
foot's position in CARLA's world (x, y; z its ground height, absent: the nearest lane's), its heading (CARLA's yaw),
the mast height, and **sensors that are rig sensor definitions** (core/rig.py; `with_defaults` fills what a
hand-written one leaves out) posed from the mast head — x along the heading, y right, z up — so raising the pole
raises every sensor. Cameras (pinhole or fisheye) and lidars; a camera always records its ground truth (semantic is
added when no label layer is asked, as the CPM reads its instance camera). Sensor faults and Falcon's sensor weather
physics are not applied to roadside sensors.

Presets (`core.v2x.PRESETS`, typical values, not products): `pole-camera` (a 1280 × 720, 90° camera tilted towards
the junction), `intersection-lidar` (32 channels, +10° to −30°, 360°, 100 m), `camera-lidar` (both). The default mast
is 6 m (a typical signal mast arm or camera pole, 5–7 m).

**Worlds ▸ Roadside units (V2X)**: the world's poles in a list beside a top-down map of its roads with each sensor's
field of view (drawn to 60 m); *Add at the junction* (an imported world: its junctions in a list, signalised first,
named after the OpenStreetMap ways that meet there), *Add at a position*; name, position, heading, mast height, CPM
rate, the visibility thresholds and the noise per pole; every change undoable. The CLI does the same
(`falcon v2x junctions|place|list|remove`, [CLI](cli.md)).

**Placement at a junction** (`core.v2x.junctions`, `corners`, `place_at_junction`): a junction of the world's
OpenDRIVE file is placed at the mean of its connecting roads' reference lines, its radius the farthest of their
points; its approaches are its incoming roads, each at the bearing of the point 10 m along it from the junction. A
pole goes on the bisector between two adjacent approaches, at the nearest point from radius + 4 m outwards (1 m steps,
up to 15 m more) that is at least 0.5 m from the nearest driving lane's edge (core.cityprops.Lanes) — a heuristic for
"on the corner, off the carriageway" — facing the centre, its camera tilted to look 10 m past it; a second pole at the
same junction takes the next corner. Towns have no OpenDRIVE file in the project, so a town's poles are placed by
position. Measured on central Bengaluru (the re-imported world): 95 junctions found; the pole of the measurement
below stands at the widest corner of *Kasturba Road × Vittal Mallya Road* (junction 98, 4 approaches, signals), 22.6 m
from its centre and 1.57 m from the nearest driving lane.

## What CARLA 0.9.16 allows

Measured on Falcon's own server (CARLA 0.9.16, the reference machine, 2026-10-30), the demo copy of the re-imported
central Bengaluru with its city, the pole of the measurement below (a 1280 × 720 camera and a 32-channel lidar, 6 m):

| question | measured (one 40 s drive, 800 steps at 20 steps/s, seed 0) |
|---|---|
| a sensor spawned with no parent (`World.spawn_actor` without `attach_to`) | stays at its world transform and delivers every synchronous tick: the pole's RGB, instance and depth cameras and its ray-cast and semantic lidars delivered all 800 frames (a missing frame fails the run) |
| the instance camera from a pole | labels actors with their ids as on the ego (a median 46 actors per frame in its objects: pedestrians, motorcycles, cars, buses, bicycles; the traffic lights' heads); like the ego's label cameras it does **not** see the generated road mesh (tagged sky): 55 % of its pixels (median) were labelled from the map's ground instead ([worlds](worlds.md#labels-in-an-imported-world-a-carla-limit-worked-around)) |
| the semantic lidar beside a static ray-cast lidar | the same rays in the same order: 800 of 800 slices matched by order; of the points inside vehicle boxes 99.9 % labelled a vehicle class |
| what the pole sees (the camera image, checked by eye) | the whole junction over the traffic: cars, buses, motorcycles with riders, walkers, the signal heads |

CARLA's own `sensor.other.v2x` and `sensor.other.v2x_custom` exist in 0.9.16 (Falcon's blueprint inventory) and were
not used (see *The messages*).

## Recording

A drive (and a scenario run, which is a drive) of a world with roadside units spawns each of their sensors
unattached at its world pose and records it like the ego's — `<rsu>.<sensor>` ids, the station frame, `cpm.jsonl` —
see [dataset format](dataset-format.md#roadside-units). Every label kind the ego's sensors have is there: RGB,
semantic, instance, range, objects (2D and 3D boxes, occlusion, track ids), lidar points with per-point tags and actor
ids, proven point for point (run.json `lidar_checks`), and on imported worlds the ground labelled from the map. A
drive started by `falcon v2x drive` puts its ego on a spawn point 50–150 m from the pole heading towards it and places
the traffic on the spawn points within 200 m of it first (`start_near`, recorded in run.json).

## The messages

Every `rate_hz` (10 Hz by default: every second step at 20 steps/s) each pole makes one CPM from its sensors' frames
of that tick (`workers/rsu.py`):

- **Perceived objects** — a vehicle or walker is perceived by a camera when at least `min_pixels` (20) pixels of its
  instance image carry the actor's id with a road-user tag (12–19), by a lidar when at least `min_points` (3) points of
  its latest full rotation hit it (the semantic lidar's object index). An occluded road user, which no pixel or point
  reaches, is never listed — that is the point. Both thresholds are heuristics; the measured benefit applies the same
  ones to the ego's sensors. The listed state is the ground truth of the tick: the 3D box centre (`position`), the
  velocity (`cartesianVelocity`), the heading (`angles.zAngle`), the yaw rate, the box's length, width and height, the
  class, the time since this pole first saw it without a gap over 1 s (`objectAge`, capped at 2047 ms), the sensors
  that saw it (`sensorIdList`), and `objectPerceptionQuality` 0–15 from the evidence (15 at 400 pixels or 40 points,
  linear below: a heuristic). With `noise` set, Gaussian errors (σ position, σ velocity; seeded by the run's seed, the
  pole and the step) are added and the 95 % confidences (`CoordinateConfidence`, `SpeedConfidence`: 1.96 σ, at least
  the 1 cm or 1 cm/s quantisation) follow; without, the confidences are the quantisation. The nearest `max_objects`
  (255, the container's root) are listed.
- **Classes** — `vehicleSubClass` passengerCar (cars, the stand-ins: their mesh's class, as in the labels), bus,
  heavyTruck, lightTruck, specialVehicle (emergency, construction), trailer; motorcycles, bicycles and pedestrians as
  VRUs (`vruSubClass` motorcyclist, bicyclist, ordinary pedestrian) — TrafficParticipantType's motorcycle, cyclist and
  pedestrian are outside ObjectClass's `vehicleSubClass` constraint. Confidence 100 %: a perfect classifier.
- **Frames and units** — East-North-Up from the pole's foot (the PerceivedObjects' frame in TS 103 324: x East,
  y North, z up): x = CARLA x − x₀, y = −(CARLA y − y₀). Positions 0.01 m, velocities 0.01 m/s, dimensions 0.1 m,
  angles 0.1° (a CartesianAngle counter-clockwise from East; a Wgs84Angle clockwise from North). The reference
  position is the foot's latitude and longitude from the map's geo-reference (CARLA's `transform_to_geolocation`).
  `referenceTime` is TimestampIts (TAI ms since 2004-01-01): `epoch` (2025-01-01T00:00:00Z) plus the simulation time,
  with the 5 leap seconds TAI − UTC gained since 2004. Every object's `measurementDeltaTime` is 0.
- **Sensor information** — per sensor its type (monovideo, lidar) and field of view as a RadialShape around its mount:
  range (a camera's is declared: `camera_range_m`, 100 m; a lidar's its own), horizontal opening clockwise from North,
  vertical opening; `shadowingApplies` true.
- **Free space** (`free_space`) — per sensor a PerceptionRegion: its field of view in `sectors` (16) slices, each cut
  at the nearest **static** occluder the sensor saw in it — buildings, walls, fences, poles, lights, signs,
  vegetation, static and dynamic props, bridges, guard rails, and vehicles that are no actor (the map's parked cars):
  camera pixels with their depth (every fourth pixel), lidar points more than 0.3 m above the ground — as RadialShapes,
  `shadowingApplies` true (what is behind a listed road user is not claimed free), the declared range where nothing was
  hit, the objects in it listed; `perceptionRegionConfidence` 101 (unavailable: there is no calibrated figure). A
  slice's range is its nearest occluder over the whole slice (conservative).
- **Management container** — `messageRateRange` the pole's rate; no segmentation.
- **Not done**: the object inclusion and generation rules of TS 103 324 (every message lists every perceived object
  at a fixed rate), segmentation over 255 objects, acceleration, correlation matrices, map positions (no MAPEM),
  free-space confidence, the radio channel, latency, loss, security headers. CARLA 0.9.16's own `sensor.other.v2x`
  (CAM-shaped messages over a path-loss model between sensors on actors) and `sensor.other.v2x_custom` (a short custom
  string) are not used: neither carries a CPM.

### The encoding, and how it was checked

`core/cpm.py` writes ITU-T X.691 unaligned PER (UPER) of the CollectivePerceptionMessage from the ASN.1 modules ETSI
publishes — forge.etsi.org `ITS/asn1/cpm_ts103324` at tag v2.1.1 and its CDD submodule `cdd_ts102894_2` at 55ae879d
(ETSI-ITS-CDD 3.1), both BSD-3-Clause, kept in `tests/data/etsi_cpm_asn1/` — in pure Python (no new dependency):
constrained whole numbers in the fewest bits, presence bitmaps and extension bits, CHOICE indices, SIZE-constrained
lengths (an extensible size beyond its root as the extension bit and a length), open types (each WrappedCpmContainer's
containerData) as a length and the inner encoding in whole octets. It encodes the containers above and refuses, naming
the component, any value outside its ASN.1 range. Two independent decoders read the same values from its bytes
(`tests/test_v2x.py`, 2026-10-30):

- **asn1tools 0.169** (MIT), compiling ETSI's files, decodes every container to the values written — with two
  patches, because asn1tools does not apply two rules: ObjectClass's `vehicleSubClass` is TrafficParticipantType
  `(unknown|passengerCar..tram|agricultural)`, a PER-visible union whose effective range is 0..14 (4 bits; asn1tools
  uses 0..255, 8 bits); and ConstraintWrappedCpmContainers applies a second, non-extensible constraint to
  WrappedCpmContainers (SIZE(1..8,…)), so its length has no extension bit (serially applied constraints take the last
  one's extensibility; asn1tools keeps the bit). Run it with `FALCON_DEVKITS_PYTHON` naming a Python with asn1tools.
- **etsi_its_messages 3.5.0** (ika, RWTH Aachen; MIT; its UPER code generated by asn1c 0.9.29 from the same ETSI
  files): its conversion node, in Falcon's bridge image, receives the bytes over ROS 2 as `udp_msgs/UdpPacket` behind
  a BTP-B header (destination port 2009) and publishes `etsi_its_cpm_ts_msgs/CollectivePerceptionMessage`; every value
  Falcon wrote is found there (`FALCON_ROS2=1`, the image rebuilt with the packages). This check found the second
  rule: the first encoder wrote the extension bit and asn1c refused every message ("Failed to decode message"); asn1c
  agrees with the X.691 reading on both rules.

The test's sample message (four objects of four classes, two sensors, two regions — one beyond RadialShapesList's
root) is frozen by its SHA-256. So the claim is exactly this: **CPMs that two independent UPER decoders built from
ETSI's ASN.1 read back to the values Falcon meant**. That is not conformance to TS 103 324: no ETSI conformance test
was run, the generation rules are not applied, and the GeoNetworking, BTP-over-GeoNetworking and security layers are
absent (the BTP-B header is there only for the conversion node).

## ROS 2 and MCAP

The stack bridge ([stack testing](stack-testing.md#topics-bridge-section-of-a-stack-config)) publishes, for each pole
of the world, at the steps it sends a CPM: `/falcon/v2x/<rsu>/cpm` (`udp_msgs/msg/UdpPacket`: BTP-B header + UPER)
and `/falcon/v2x/<rsu>/cpm_json` (`std_msgs/msg/String`: the CPM and Falcon's extras as JSON), stamped with the
frame's simulation time; the stack run writes `cpm.jsonl` too. A stack run spawns only the label sensors the CPM
needs (instance and depth cameras, semantic lidars). To get typed ETSI messages, run etsi_its_messages' conversion node
beside the bridge — `falcon v2x converter RSU` prints its command (in the bridge image: `/falcon/v2x/<rsu>/cpm` in,
`/falcon/v2x/<rsu>/cpm_ts` out). The bridge image (`falcon stack image`) now installs `ros-humble-udp-msgs`,
`ros-humble-etsi-its-conversion` and `ros-humble-etsi-its-cpm-ts-msgs` from the ROS apt repository (MIT licences;
the image grew from 1.15 to 1.42 GB); a node in an older image says it cannot publish the UdpPacket topic and the run
leaves it out, noted. `falcon export … --format mcap` writes both topics from `cpm.jsonl`; the roadside sensors'
images and points are not in the bag (the rig's are).

## Measured: what the RSU adds to the ego's view

Measured on Falcon's own server (CARLA 0.9.16, the reference machine, 2026-10-30) through `falcon v2x drive` and
`falcon v2x benefit`, on the demo copy of the re-imported central Bengaluru (world-8c0095b7: left-hand traffic, its
city of stand-in buildings, trees and lamps on, ClearNoon). The pole: `camera-lidar` at 6 m on the widest corner of
*Kasturba Road × Vittal Mallya Road* (signalised, 4 approaches). The ego: CARLA's Lincoln MKZ with the *Camera and
lidar* rig — a 1280 × 720, 90° front camera at 1.6 m and a 32-channel, 80 m lidar at 2.2 m — on the autopilot,
started 50–150 m from the pole heading for it. Traffic: the Indian mixed profile with 80 vehicles (fleet weights
motorcycles 0.45, cars 0.26, microlino stand-ins 0.12, buses 0.06, trucks 0.04, vans 0.03, bicycles 0.04) and 40
walkers, placed on the spawn points within 200 m of the pole first. Five drives (seeds 0–4) of 40 s at 20 steps/s,
every second step recorded (10 Hz, the CPM period). Road users counted: every vehicle and walker within **50 m** of
the ego (the ego left out), at the 400 CPM steps of each drive; seen by the ego: ≥ 20 instance pixels in its camera
or ≥ 3 points in its lidar's latest sweep; by the RSU: listed in its CPM of that step (the same thresholds on the
pole's sensors). "Near the pole": the ego within 60 m of it. The machine was shared (another CARLA server and GPU
jobs ran): 1.0–1.7 steps/s, which changes nothing measured here (synchronous mode).

Share of the road users within 50 m that the ego knows about, its own sensors → its sensors and the CPMs:

| seed | road-user steps | all | vehicles | VRUs (walkers, two-wheelers) | near the pole | the drive |
|---|---|---|---|---|---|---|
| 0 | 6 661 | 90.1 → 97.3 % | 95.5 → 98.7 % | 87.1 → 96.5 % | 90.0 → 97.4 % | through the junction (432 m, 3 m from the pole) |
| 1 | 10 088 | 83.2 → 97.8 % | 98.8 → 100 % | 69.8 → 96.0 % | 83.2 → 97.8 % | queued at the junction (38 m) |
| 2 | 8 794 | 79.6 → 95.6 % | 97.7 → 100 % | 73.3 → 94.1 % | 85.5 → 98.5 % | queued (27 m) |
| 3 | 7 460 | 80.6 → 96.3 % | 88.3 → 98.1 % | 71.4 → 94.1 % | 80.4 → 96.2 % | through the junction (401 m, 7 m from the pole) |
| 4 | 13 544 | 49.9 → 78.7 % | 89.1 → 99.5 % | 33.3 → 69.9 % | — (never within 60 m) | queued 65–84 m away, behind buses (118 m) |
| **pooled** | 46 547 | **73.4 → 91.6 %** (+18.1 points; per drive +7.2 to +28.8) | 93.5 → 99.3 % | **61.4 → 86.9 %** (+25.5) | **84.5 → 97.5 %** (+13.1; 30 337) | |

Large vehicles (buses, trucks: 3 713 road-user steps) were almost always seen by the ego already (99.1 → 100 %): they
are what hides the others. The gain is in the small road users — motorcyclists, cyclists and walkers behind buses,
cars and buildings. The same drives counted with only the ego's lidar: 69.5 → 90.6 % (VRUs 55.4 → 85.5 %); with only
its front camera (a 90° view, so most of the loss is field of view, not occlusion): 30.7 → 88.7 %.

**Time to first detection out of occlusion** (episodes: a road user within 50 m unseen by the ego for at least 0.5 s,
ending when the ego sees it): 124 episodes over the five drives; in 116 the CPM had already named the road user,
on average **2.4 s earlier** (median 1.1 s when earlier; the longest 28.5 s, a walker the pole saw from the first step and the queued ego only
after 28.5 s, seed 4).
59 road users were never seen by the ego while within 50 m; 45 of them were in the CPMs.

What the messages cost: 400 CPMs per drive, 2.0–2.4 kB on average (UPER; the largest 3.2 kB) for 30–66 objects (5th–95th percentile, median 46) —
above what one radio packet carries; TS 103 324's object inclusion rules and segmentation, not applied here, would
shrink and split them. Of the objects listed, 41 % were seen by the pole's camera only, 25 % by its lidar only, 35 %
by both.

How to read it: an upper bound for this pole — perfect detection and classification behind the sensors' real
occlusion, every message delivered at once — at one junction, in five 40 s drives with traffic packed around it (in
three of them the ego queued in it). It says where a roadside unit helps (VRUs hidden by larger vehicles near a busy
junction), not how much a deployed RSU with a real detector and radio would.

## Limits

- Ground truth behind real geometry: the visibility rule is the sensors' (instance pixels, lidar points), the state
  is exact (or with set Gaussian noise), the class perfect — no detector, no tracker, no false positives, no ID
  switches. The benefit measured is therefore an upper bound of what a CPM from this pole could add.
- No radio: every message arrives, at once. No ETSI generation rules (rates, object inclusion), no segmentation.
- The thresholds (20 pixels, 3 points), the quality scale and the placement corner are heuristics.
- A roadside camera's range in the CPM is declared, not measured; free space is per slice (16 by default), its nearest
  occluder, and has no confidence.
- Static sensors only: no pole sway, no calibration error of the pole; faults and weather physics are not applied to
  them.
- Towns: placement by position only (their OpenDRIVE is the server's).

## Tests

`tests/test_v2x.py`: the encoding (round trips, the ITS header, the extension of an extensible size, ranges refused
by name, frozen bytes; asn1tools on ETSI's ASN.1 when installed; the asn1c decoder with `FALCON_ROS2=1`), the frames
and presets, junction placement on a converted world, the schema and the drive config, the spawn order; a drive with a
pole against the fake carla (whose sensors may stand in the world, as CARLA's): the roadside sensors' files and labels,
no ego class from a roadside camera, every CPM decoding to its JSON, every listed object seen by enough pixels or
points and no camera evidence for an object without pixels in the recorded instance image, the ground truth in the
ENU frame; the benefit and decode commands; the MCAP export; the stack bridge publishing the CPMs through the node's
logic; the Worlds page's section (a town and an OpenDRIVE world).
