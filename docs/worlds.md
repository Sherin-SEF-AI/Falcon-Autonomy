# Worlds, places and traffic

## CARLA's towns
A town world names a CARLA map (Town01–05, Town10HD and their `_Opt` variants in the base package; Tools ▸ Probe lists
the server's own) with its weather and time of day. Every one of CARLA's towns is **right-hand traffic**.

## Importing a real place
Worlds ▸ **Import a place…**:

1. **The source** — an `.osm` or `.osm.pbf` file (offline first; Falcon reads both itself, with no extra library).
   A place-name search (OpenStreetMap Nominatim) only finds the area, and only when *Settings ▸ Geocoding* allows the
   network; the roads always come from a file.
2. **The area** — drag a rectangle over the file's roads, or give a centre and a side (1000 m: 1 km²). Ways crossing
   the box are kept whole. Files over 200 MB are not previewed whole: give the centre and side.
3. **The converter's settings** — default lane width, traffic lights where OSM has them or at every junction, which
   road types are converted, and *Left-hand traffic*.
4. **Convert** (a background job in CARLA's Python; no server needed). The preview shows the converted road graph,
   coloured by driving lanes per direction; the report gives the road length, junctions, signals, the lanes histogram,
   the handedness, the **lane check against OSM** and what an OpenDRIVE world does not contain.
5. **Add as a world.** The world's files stay in the workspace (`worlds/<id>/`: `area.osm`, the converter's
   `raw.xodr`, the loaded `world.xodr`, `graph.json`, `import.json`). Its mesh settings (vertex distance, wall
   height, extra width, smoothing) are used when a drive or live session generates it on the server.

**The projection.** Falcon converts with a transverse Mercator centred on the area (`+proj=tmerc +lat_0 +lon_0 …`,
`center_map` off), so the OpenDRIVE's x and y are metres east and north of the centre and its geoReference is the
area's. CARLA's default (`+proj=tmerc`, centred on 0°E) stretched central Bengaluru about 3× east-west and 4×
north-south.

**A crash fixed before loading.** CARLA 0.9.16's world generation ends the server (signal 11, *s <= road->GetLength()*)
when an object lies past its road's end, and the converter writes some crosswalks a few millimetres past it (measured:
one of 19 crosswalks, 3.5 mm out, in 1 km² of central Bengaluru). Falcon moves such positions onto the road when they
are within 5 cm and lists each in the report; anything further out is refused rather than moved.

**The lane check.** A seeded sample of the area's named, lane-tagged OSM ways is matched to same-named converted roads
running within 8 m of them; each direction's driving-lane count is compared with what the OSM tags say. The report
gives the agreement over the comparable ways and lists the others. It also checks **where** the lanes lie: every 2 m
of each matched road, the middle of its driving lanes in each direction against where OSM puts it (on the way for a
one-way way; beside it on the driving side, half that direction's width out, for a two-way way). It reports the
median and 95th percentile error and the ways whose lanes are mostly off by more than 0.75 of a lane width
(*misplaced*). The converter's own placement counts too: in central Bengaluru one of 23 checked ways is misplaced
before any conversion (a one-way road the converter puts beside its way rather than on it).

## Left-hand traffic: what was verified
Measured on CARLA 0.9.16 (the reference machine, 2026-10-28), with a straight two-lane OpenDRIVE road loaded by
`generate_opendrive_world`:

| road `rule` | lane +1 (left of the reference line) | lane −1 (right) | traffic-manager cars |
|---|---|---|---|
| `LHT` | waypoints along +s | along −s | drive in lane +1 along +s and lane −1 along −s: **on the left** |
| `RHT` or none | along −s | along +s | on the right |

So **CARLA 0.9.16 honours OpenDRIVE's `rule="LHT"`** in its map (waypoint directions) and its traffic manager.

**CARLA's OSM converter (`carla.Osm2Odr`, SUMO's netconvert inside) does not**: its settings have no handedness
option, and its OpenDRIVE has no `rule` — every road is right-hand. Each travel direction of a street is its own road
with right lanes only. Where the reference line lies differs (measured on central Bengaluru): the two directions of a
two-way street share one, the street's centre (the OSM way); a one-way street — or one carriageway of a dual
carriageway — is one road whose reference line is its carriageway's **left edge**, with the way down its middle.

**What Falcon does**: with *Left-hand traffic* on (the default for an Indian place), Falcon converts the converter's
output: every road's lanes are mirrored (lane k → −k), every road gets `rule="LHT"`, lane offsets, lane links,
junction lane links and signal positions are mirrored with them. Each road keeps its travel direction; only the side
changes. The mirror line is the line the road's OSM way runs along: the reference line for a two-way street (its
directions swap sides), the carriageway's centre for a one-way road (it stays where it is: a laneOffset does it).
Without the OSM ways (a file converted on its own) a road that shares its reference line with an opposite road is
mirrored there and a lone one-way road about its carriageway's centre. Junction connecting roads then get a smooth
(cubic) laneOffset that meets the lanes they link at both ends. After loading, a verification run spawns traffic and
checks which side it drives on; the report says so.

Before this (Falcon 2.1.0 and earlier) every road was mirrored across its reference line, which moved each one-way
road a whole carriageway width sideways: in central Bengaluru 36 of 245 roads (4.2 of 19.0 km) lay 3.35–13.4 m
(median 6.7 m) off their way, and 11 of the 23 checked ways were misplaced; on a highway mapped as two one-way ways a
dashcam's GPS fell on a lane 0.7 % of the time against 59.9 % now. A world imported before needs importing again
to get the corrected lanes: Worlds ▸ *Re-import with the current converter…* does it into a new world beside the old
one (below).

**Traffic lights in a left-hand world.** Mirroring a road moves the lanes a signal applies to to the other side, so
the signal's orientation (`+` / `-`) must flip with them. Until this version it was kept, and CARLA 0.9.16 then gave
every traffic light of a left-hand import **no lane to control** (`get_affected_lane_waypoints()` empty): traffic-
manager cars drove through red. Measured on a real server: central Bengaluru's 7 lights controlled 0 lanes each as
imported by 2.1.0, and 4, 1, 4, 5, 3, 5 and 5 lanes after a re-import; on a synthetic crossroads with every light
held red, a car 80 m away drove through at 29 km/h before the fix and stopped 10.4 m short of the crossing's centre
after it — as it does in the same world converted right-hand. Re-import a left-hand world to get working lights.

| central Bengaluru, 1 km² (245 roads, 390 junction roads) | the converter (right-hand) | mirrored only (2.1.0) | now |
|---|---|---|---|
| lane positions against OSM, 23 checked ways: 95th percentile; misplaced | 0.58 m; 1 | 10.05 m; 11 | 0.58 m; 1 |
| linked lanes that do not meet (> 0.5 m), of 1136 links; 95th percentile gap | 150; 0.77 m | 202; 6.7 m | 150; 2.25 m |

The remaining gaps are where the converter's junction roads meet a road at an angle (a kink): mirrored about two
lines at an angle, a lane's ends move apart **along** the road, which a laneOffset (sideways only) cannot close —
up to 5 m in central Bengaluru. The converter's own file has such gaps too (whole-city export: 822 of 14 168 links
over 0.5 m before conversion, 820 after; mirroring only: 1214). The whole export (7554 roads) converts in 4–5 s.

**The limit**, stated in the import's report: on approaches with two or more lanes per direction, which lane turns
where at a junction is the right-hand converter's assignment mirrored (the inner lane keeps the turns it had), not a
true left-hand assignment. Single-lane approaches — most streets in an OSM extract — are unaffected. A fully
left-hand conversion needs SUMO's own `netconvert --lefthand`, which Falcon does not run in this version.

## Indian mixed traffic
The *Indian mixed traffic (Bengaluru)* profile: left-hand; fleet weights — motorcycles 0.45, bicycles 0.04, cars
0.26, vans 0.03, buses 0.06, trucks 0.04, autorickshaw substitute 0.12 (heuristic: two-wheelers are about two thirds
of vehicles registered in Bengaluru; not an on-road count); a 1 m gap to the vehicle ahead, a ±25 % per-vehicle speed
spread, 15 % ignoring lights, 25 % ignoring signs, 10 % random lane changes each way, 35 % of walkers crossing
anywhere. Every value is editable on the Worlds page.

**Autorickshaws** are not in CARLA 0.9.16. The profile's stand-in is `vehicle.micro.microlino`, the vehicle nearest in
measured size, driven with a three-wheeler's top speed and acceleration and labelled *autorickshaw (stand-in)*
(`subclass` in the labels). [Indian road assets](indian-assets.md) gives the measurements and the source-build path
to a real one (then name its blueprint in the profile's fleet: `blueprint:vehicle.<make>.<model>`).

**Road features**: an imported world can carry speed breakers as real geometry (elevation profiles the vehicle physics
drives over), placed from OpenStreetMap's `traffic_calming` tags at import and from the Worlds page (*Road
features*); such a world's mesh is generated at a 0.25 m vertex distance. Puddles (`precipitation_deposits`) are not
drawn on an imported world ([Indian road assets](indian-assets.md)).

**Lateral filtering**: the traffic manager keeps every vehicle on its lane's centre. A profile's *Lateral filtering*
(Worlds page, traffic profile; `lateral_filtering` in the project) gives its two-wheelers — and, when asked, the
autorickshaw stand-ins — a lateral position of their own: the lane line, between vehicles, the kerb side, never an
oncoming lane. [Indian road assets](indian-assets.md#lateral-filtering) has the design, the measurements on this
world and the limits.

## Speed limits and speed breakers

CARLA 0.9.16's traffic manager drives every autopilot vehicle at `Vehicle.get_speed_limit()`: the limit of the last
speed sign it passed, else 30 km/h. A generated OpenDRIVE world has **no speed signs**, and the traffic manager reads
neither the file's `<speed>` records nor its speed-limit signals (measured above) — so before 2.3 every vehicle drove
about 30 km/h on every road of an imported place, a 20 km/h lane and a 100 km/h highway alike.

On an imported world Falcon now gives each autopilot vehicle — the traffic, an autopilot ego, the
[stand-ins](indian-assets.md) — its desired speed itself, with `TrafficManager.set_desired_speed`, re-read every
0.25 s of simulated time (`workers/speedcontrol.py`, `core/speedlimits.py`):

1. **The lane's limit** from the file: the lane's `<speed>` record at its `s` (what CARLA's converter writes from OSM
   `maxspeed` or the road type's default, in m/s), else the road's `<type><speed>`, else CARLA's own
   `get_speed_limit()` (counted in run.json as `carla`).
2. **The vehicle's difference**: the traffic profile's *Speed below the limit* with its per-vehicle spread (the Indian
   profile: 10 % faster ± 25 %), as the traffic manager would apply it; a stand-in is capped at its class's top speed.
3. **What lies ahead** along its lane (waypoints every 5 m, as far as slowing to 10 km/h at 2 m/s² needs, at most
   150 m; at a branch the one that keeps the road, else the straightest): a lane with a lower limit, or a **speed
   breaker** (the import's `features.json`, crossed at its crossing speed — 25 km/h for an IRC:99 hump), lowers the
   target to the speed from which a steady 2 m/s² reaches the slower speed there (√(v² + 2·a·d), less one update's
   travel and 5 m). Drivers slow before the sign, not after it; a breaker's speed is held until 3 m past it.
   *Heuristic*: 2 m/s² is a comfortable deceleration (AASHTO's stopping sight distance assumes 3.4 m/s²); the traffic
   manager picks its own route at junctions, so the look-ahead's branch is a guess.

A world entry's `speed_limits: "carla"` (project file) turns it off — the traffic manager's own 30 km/h. CARLA's towns
have speed signs and are left to the traffic manager. run.json `traffic.speed_source` records the mode, the limits in
the file, how many vehicles of each role were held, the samples by limit source, the breaker approaches and the
updates slowed by a limit or a breaker ahead.

**Measured** (CARLA 0.9.16, Falcon's own server, 2026-10-29; speeds of moving vehicles, junction roads and breaker
approaches left out; drives of 90 s and 120 s at 20 Hz, the first 10 s left out):

*A synthetic right-hand ring* (radius 250 m, three sectors tagged `maxspeed` 20, 50 and 80, two IRC humps on the 80
sector, 5 cars and the ego; few spawn points on a ring):

| sector | CARLA's own (median, max) | Falcon's limits (median, p85, max) |
|---|---|---|
| 20 km/h | 29.0, 29.3 | 19.3, 19.4, 22.5 |
| 50 km/h | 29.0, 29.1 | 48.3, 48.5, 48.5 |
| 80 km/h | 28.9, 29.1 | 63.8, 78.2, 78.4 (much of the sector is spent slowing for its humps and the 20 sector after it) |

Without the look-ahead a car entered the 20 km/h sector at 78 km/h and slowed only after it (the 20 sector's maximum
was 78.2). Approaching a hump on the 80 sector (three passes, the ego's among them): 47–49 km/h 40 m before it, 37
at 20 m, 25–29 at 10 m, **24.0–24.6 km/h on it**; with CARLA's own speeds they crossed at 29 km/h.

*The re-imported central Bengaluru* (left-hand, the Indian profile, 60 vehicles and the ego, all held by the speed
control; limits in the file from 12 to 100 km/h, the slow ones SUMO's junction connections). In both modes alike
vehicles leave the 1 km box at its dead ends and are removed (61 → 26 alive after 120 s with Falcon's limits, 61 → 19
with CARLA's own), so the later samples come from fewer vehicles:

| lane limit | CARLA's own (median, p85, max) | Falcon's limits (median, p85, max) |
|---|---|---|
| 20 km/h | 32.0, 36.5, 55.0 | 21.4, 24.5, 26.6 |
| 25 km/h | 30.1, 36.3, 95.3 | 24.9, 29.3, 33.0 |
| 50 km/h | 28.3, 36.6, 110.8 | 31.2, 45.0, 65.6 |
| 80 km/h | 30.4, 38.3, 39.4 | 38.3, 79.1, 102.4 |
| 100 km/h | 28.6, 35.2, 38.4 | 45.7, 89.6, 111.5 |

Above the limit is by design: the Indian profile drives 10 % faster ± 25 %. Medians stay below the limits of the fast
roads: they are short in a 1 km box, with signals, queues and slower junctions ahead (3029 updates were lowered for a
lower limit ahead). The two breakers of the area saw one traffic pass (28.7 → 23.8 km/h on it) and the ego's (24 km/h,
already on a 25 km/h lane). The cost: 23.3 s for 2400 steps against 22.6 s without (27 vehicles held).

**Limits**: the traffic manager accelerates and brakes by its own controller, so a target is reached with its lag;
two-wheelers and bicycles are held like cars (no class of their own); walkers are untouched; a breaker without a crossing speed (a pothole) does
not slow anyone; a scenario's own actors keep the speeds the scenario sets; Live's cutting-in car keeps its own.

## A standalone OpenDRIVE world: what it is
`generate_opendrive_world` builds **roads, lanes, sidewalks and junctions** as meshes from the OpenDRIVE file — no
buildings, trees, real textures, road paint beyond lane marks, or props. Falcon says so on the Worlds page.

CARLA cannot build a new mesh at runtime (an extruded footprint, say), but `static.prop.mesh` places any mesh
already cooked into the package — the towns' own buildings, trees and lamps among them. Falcon uses that to put a
city around the roads (next section): stand-ins, of about the right size, not the real buildings.

## A city around the roads
Worlds ▸ **City around the roads** (`core.cityprops`) dresses an imported world from an OpenStreetMap file — the
import's own, or another with the buildings a roads-only download lacks (a regional `.osm.pbf` extract; read, never
changed). A new import builds it from its source at once; an older world gets it from *Build the city*. It is placed
in every run of the world while *In runs* is on, recorded in `run.json` (`city`: the plan's SHA-256, how many spawned,
the time it took) and its plan's hash is in the drive's config, so it is part of the cache key. Off, the world is the
roads alone, as before.

**What CARLA 0.9.16 allows** (measured on Falcon's own server, 2026-10-29):

| question | measured |
|---|---|
| `generate_opendrive_world` | roads, sidewalks and junctions only; its parameters (vertex distance, wall height, additional width, smoothing, mesh visibility, pedestrian navigation) add nothing else |
| the library's props | 98 `static.prop.*` (benches, bins, bus stops, kiosks, an ATM, plant pots, a coconut palm, a cypress…); no building |
| `static.prop.mesh` with a package path | 485 of 488 candidate paths under `Static/Building`, `Static/Vegetation` and `Static/Pole` spawn with a real box (3 spawn empty); 92 whole buildings among them (not façade pieces, standing on their origin, 4–80 m), 15 trees and 4 lamps chosen for the city |
| labels | buildings tag 3 (*building*), trees and bushes 9 (*vegetation*), lamps 6 (*pole*) in the semantic and instance cameras, the instance id being the actor's; bus stops and benches 20 (*static*), bins 21 (*dynamic*), `kiosk_01` 3 |
| collision | `cast_ray` hits the buildings and lamps (label *Buildings*, *Poles*); it missed the palm and oak trunks along their boxes' centre lines, so whether trees collide is not established |
| cost | 3341 actors spawned in one synchronous batch in 1.18 s the first time, 0.11–0.12 s after (assets loaded); a recorded drive (one 1280 × 720 camera with labels, 15 vehicles, 10 walkers) ran at 9.2 steps/s against 10.7 without (−14 %, four seeds each) |
| ground | no flat mesh with a plausible ground surface: `SM_Plane` is an upright debug plane (red), `WaterPlane` draws water, `Big_Plane` flowers, `SM_town03_SM_terrain02` is a 5 m-high terrain chunk; beside the roads the world stays CARLA's empty grey |

**How it is placed.**

- *Buildings.* Each closed OSM footprint (multipolygon buildings are relations, which the reader skips) is reduced
  to its minimum-area rectangle; its height is OSM's `height`, else `building:levels` × 3.2 m, else a typical number
  of storeys for its `building` type (*typical*, said so in the summary). The catalogue's building whose box,
  turned either way and uniformly scaled within 0.6–1.6×, is nearest in size (the RMS of the relative differences of
  length, width and height — `core.assets.size_score`) is chosen, among those within 0.03 of the best by the seed;
  the scaled box may exceed the footprint's rectangle by 0.3 m a side. It may not come within 0.3 m of any lane of
  the map (sidewalks included) — it is moved up to 3 m away from the nearest one, then shrunk to 90, 80, 70 % — nor
  overlap a building already placed (largest footprints first). Its façade's side is unknown: it faces either way
  along the length, by the seed. **It is a stand-in**: the style, the façade, the roof and an L-shaped footprint's
  shape are the town mesh's.
- *Trees*: OSM's `natural=tree`, and positions spread over parks, gardens, grass, woods and tree rows (a seeded
  Poisson-disc sample, 9 m apart in parks, 6.3 m in woods, 14.4 m on grass), off carriageways, sidewalks and
  buildings; at most 2000 spread trees, those nearest a road kept. The species are stand-ins (palms, broad-leaved
  trees of the package).
- *Lamps*: OSM's `highway=street_lamp` (a park lamp where no road is within 25 m) and one every 30 m along each
  sidewalk outside junctions, the arm over the carriageway (procedural, counted apart).
- *Street furniture*: bus stops, benches, bins, ATMs, post boxes, vending machines and drinking water at OSM's
  points, as the library's props; one mapped on the carriageway is moved onto the nearest sidewalk.

**Measured on central Bengaluru** (the re-imported world, a scratch copy; the buildings from a regional
OpenStreetMap extract on this machine, 1123 footprints within the box plus 150 m, read in 348 s; planned in 1.1 s):
858 buildings placed (125 left out as they would cover a lane or sidewalk, 134 as they would overlap another, 5
fitting no stand-in, 1 under 12 m²; 104 moved off a lane by up to 3 m); size closeness median 0.264; heights from
`height` 10, from levels 98, typical 750. 2026 trees (86 OSM's, 1940 spread; 2081 left out over the cap), 329 lamps
(26 OSM's, 303 procedural), 21 bus stops, 60 benches, 34 bins and 13 other props — 3341 actors. Over four 4 s drives
each, city off and on, the share of pixels: building 0 → 11.1 %, vegetation 0 → 16.4 %, pole 0 → 0.7 %, sky
55.6 → 30.2 %, road 22.1 → 22.0 %; the ground relabelling still named 31 % of each frame. The labelled objects (vehicles,
walkers) fell from 3297 to 742 over the same frames, with the same traffic placed: buildings and trees now hide
what is behind them.

**Limits.** Stand-ins, not the real buildings or trees; boxes for L-shaped footprints; no façade orientation; no
ground; trees' collision not established; the props are destroyed and spawned again with every run. **The lamps give
no light**: measured at night (sun −30°) on the re-imported central Bengaluru, the same view with each of the four
street-lamp meshes and the package's one light-named prop (`static.prop.gardenlamp`) beside the road had a mean
brightness of 0.147–0.161 against 0.162 without (out of 255; a car with its low beams on: 1.033), and the light
manager's count stayed 0. CARLA 0.9.16's Python API cannot add a light: `LightManager` only switches, dims and colours
the lights a map was built with (an imported world has none), and no spawnable blueprint carries one — a vehicle's
own lamps are the only light that can be put into an imported world at runtime. Its streets are dark at night. A world's city is
placed in every worker that loads the world: drives, scenarios, live sessions, replays, dashcam routes, stack testing
and — since their configs carry the world's settings (`core.drive.world_keys`: the city, `speed_limits`,
`keep_traffic`, `vehicle_lights`) — the learning environment, adversarial traffic and hardware-in-the-loop.

## Roads that end: the edge of an imported area

An imported area is cut from a larger map: ways crossing its boundary are kept whole, so some roads run a little past
the edge and then simply stop. Past their end there is no ground — an OpenDRIVE world is only its roads — and CARLA's
autopilot does not stop at a dead end: it drives on, off the road mesh, and falls until Unreal destroys the vehicle
below the world's kill depth. Measured on a 1 km import of central Bengaluru: an ego on the autopilot in Live left a
dead end 542 m from the centre after 31 s; the view showed a grey void and a speed climbing by 35 km/h every second
(gravity), and the session ended with "trying to operate on a destroyed actor".

Falcon's **road-end guard** (workers/scene.py `RoadEndGuard`) watches the ego in Live sessions and drives: when it is
on no driving lane and sinking (more than 1 m below its last height on a lane, or falling faster than 3 m/s), it is
put back on the road at a spawn point near the middle of the area and at least 60 m from where it left, stopped. Live
logs it ("the road ended at …"); a recorded drive marks the step in frames.jsonl (`"teleport"`) and run.json's notes,
since the pose jumps there. Scenario runs are not guarded (the director places its ego on the scenario's road).
CARLA's own towns are closed and never trigger the guard.

The traffic manager removes an ego on the autopilot too, the way it removes traffic (below): well before it reaches a
dead end. So on a lane the guard also looks ahead with the traffic keeper's dead-end check (below) and puts the ego
back the same way before it gets there, at a spawn point with no dead end within 40 m ahead, its autopilot restarted
(`"teleport"` with `"cause": "dead_end_ahead"`; a fall is `"fell"`). Measured on the demo copy of the re-imported
central Bengaluru with the city, the default rig and Indian mixed traffic, 60 s drives: with the original look-ahead
(2 s of travel, at least 12 m) the ego was removed in 2 of 8 drives, at 26–30 m before a dead end at city speed and
48 m before one at 16.5 m/s; with the current one (3 s, at least 40 m) in none of 12 (one put back ahead of a dead
end, two after a fall). Should the server destroy the ego all the same, the run fails saying so and where it was last
seen (a failed run keeps only its run.json).

**Traffic** is lost at an imported area's edges too. Measured on a scratch copy of the re-imported central Bengaluru
(Indian mixed traffic, 60 vehicles, CARLA 0.9.16), two things take it:

- **The traffic manager removes it.** Its OSM mode — on by default in 0.9.16 — destroys a vehicle as soon as the path
  it plans meets a dead end of its road graph (54 vehicles, 30 s: 41 left with the mode on, 54 with it off) — well
  before the vehicle gets there: the ego was removed 26–30 m before a dead end at city speed and 48 m before one at
  16.5 m/s (its planned path cannot be read: `get_all_actions` gives the next action only, a few metres ahead). That
  also explains most of an earlier count, 41 of 64 vehicles removed in 180 s last seen more than 40 m from a lane end
  without a successor; whether the traffic manager's graph has dead ends the map lacks as well is not settled. With
  the mode off the vehicles are not removed but drive on past the end of their path, off the road mesh, and fall
  (39 of 54 in 150 s).
- **Holes in the road mesh**, where a junction's connecting road does not meet its lanes (150 of 1136 lane links more
  than 0.5 m apart, see above): a vehicle sinks through the road where the map still has a lane.

Without help, 60 vehicles became 14–17 after 150 s. Falcon's **traffic keeper** (workers/trafficguard.py
`TrafficKeeper`) keeps the count. Every 0.25 s of simulated time it looks at each traffic vehicle:

- **removed** (by the traffic manager or the server): a vehicle of the same blueprint is spawned in its place and set
  up as the traffic profile set the first (behaviour, speed difference, class, road speed);
- **falling** (1 m below its lane, or faster than 3 m/s) or **off the road network** for 2 s: moved at once;
- **heading for a dead end of the map** within 3 s of travel (40–75 m: longer than the traffic manager's reach
  above), to the traffic manager's next action and then the lane straight on — a lane end with no successor and no driving lane 6 m beyond it (a lane that merely ends
  beside a road going on is not one): moved, unless the ego might see it go (within 50 m, or within 120 m and 60° of
  its heading — a heuristic for the rig's cameras); then it comes off the autopilot, stops with full brakes like a car
  at a closed road, and is moved once out of view.

A move or a replacement takes a spawn point in a seeded order, at least 60 m from where the vehicle was, out of the
ego's view, with no vehicle within 10 m and no dead end within 40 m ahead; the vehicle is placed before the next tick
and its autopilot restarted there. Every move is in frames.jsonl (`"traffic_respawns"`) and run.json
(`traffic.keeper`: counts, the longest hold, the first 200 events). A world can opt out with `keep_traffic: false`.
Measured with the 40 m look-ahead (the 12 drives above, 60 vehicles, 60 s): 60 alive at the end of every drive;
per drive 16 caught before a dead end (mean), 12 caught falling, 7 removed by the traffic manager and replaced (4–13),
against 18–27 removed and 0–1 caught before a dead end with the old 12 m look-ahead.

| 180 s, 60 vehicles, seed 7 | alive at 30 / 60 / 90 / 120 / 150 s (min–max) | at the end | moves |
|---|---|---|---|
| `keep_traffic: false` | 35–47 / 27–35 / 23–27 / 17–23 / 14–17 | — | — |
| the keeper | 58–60 / 59–60 / 59–60 / 59–60 / 59–60 | 60 | 64 replaced, 31 caught falling |

None of the 95 vehicles moved or replaced vanished or appeared within 120 m and 45° of the ego's heading (its front
camera's view). The removals themselves are the traffic manager's: a vehicle it destroys in view vanishes there
(none did in this run, but nothing prevents it). Tried and left out: the OSM mode off, with a move wherever the
planned path runs out — `get_all_actions` gives the next actions, not the end of the path, and that rule moved
vehicles 3882 times in 180 s, nearly all needlessly.

## Editing the map
Worlds ▸ **Edit the map…** (an imported world): the converted lanes are drawn under the OpenStreetMap ways they came
from, with the junctions, traffic signals and speed breakers; wheel or *Zoom* zooms, a drag pans, *Fit* (F) shows the
whole area. Click a road or a junction to see what can be changed:

| edit | on | what changes |
|---|---|---|
| road type | a road | OSM `highway` (one of the types the world converts): the converter's default lanes and speed for the type |
| direction | a road | `oneway`: two-way, one-way along the way, one-way against it (the lanes all go one way) |
| lanes forward / backward | a road | `lanes`, `lanes:forward`, `lanes:backward` (at least one lane each way on a two-way road; at most 6) |
| speed limit | a road | `maxspeed` (5–150 km/h, or the type's default) |
| remove | a road (Delete) | the way is left out |
| traffic signals | a junction | `highway=traffic_signals` on the node (only at a node two or more roads share) |
| speed breaker | *Add a speed breaker*, then click a road | a node on the way at that point, `traffic_calming` = hump, bump or table — placed as an elevation profile by the [road features](indian-assets.md) step like any tagged breaker |
| remove a breaker | a breaker | its `traffic_calming` tag (or the edit that placed it) |

**Edits change what the converter reads, not what it writes.** Each is a project edit (undoable) to the world's
`osm.edits` list (`core.mapedit`, `falcon.map_edits/1`); the list is applied in order to the area's OSM data before
the conversion, deterministically (new breaker nodes are numbered after the data's largest id, in the list's order),
so lanes, junctions, lane links and signals all come from the same pipeline as any import, and a re-import keeps the
edits. An edit that would leave a road without a lane is refused with the reason; one whose road the area no longer
has is skipped and marked. Removing a road that is the only link to others says which roads it cuts off before
anything is converted (vehicles there could not leave; the server removes traffic-manager cars at dead ends).

**Apply the edits** converts the area again with the list (a background job, CARLA's Python, no server) into a new
folder of the world (`worlds/<id>/edit-<hash>/`, with `area.source.osm`, the area as read, unedited) and points the
world at it — one more project edit: undo returns to the previous files, which are kept. Applying also converts with
the current converter (so it carries this version's fixes). The checks shown after: the edited roads' lane counts
against what was asked (every edited way, not a sample), the lane check's counts and positions, the lane links that
do not meet, the driving side, the network's roads and signals, the speed breakers placed. `import.json` records the
list (`map_edits`: applied, skipped and why, its SHA-256, connectivity); the world entry keeps `applied_edits`.

**What CARLA 0.9.16 does with each edit** (measured on a real server, a synthetic crossroads converted by
`carla.Osm2Odr`, left-hand, 2026-10-29):

| edit | the converter (carla.Osm2Odr) | CARLA |
|---|---|---|
| lanes, direction, road type, remove | built as asked (lanes 2 + 1 asked, 2 + 1 found) | driven as built |
| speed limit | written as the lanes' `<speed max>` records (20 km/h → 5.56 m/s) | **ignored by CARLA**: the traffic manager drives to the vehicle's limit, which stays 30 km/h (a car drove at 28.9 km/h on a road edited to 20). A speed-limit signal (OpenDRIVE type 274) becomes a map landmark, but a generated world spawns no speed-limit sign, so the limit does not change either (tried at 20 and 40 km/h). **Falcon applies it itself** (below, *Speed limits and speed breakers*) |
| traffic signals | a traffic-light signal per approach and a controller (one junction: 4 signals, 8 controllers) | 4 traffic-light actors (green and red seen) that control their lanes with this version's left-hand fix; a car stops at red |
| speed breaker | — (placed by Falcon as an elevation profile) | driven over: the car rose 4–5 cm crossing an IRC hump at 29 km/h (mesh at 0.25 m) |

So CARLA itself ignores a speed limit edited on the map; Falcon reads it from the file and gives every vehicle its
road's limit ([below](#speed-limits-and-speed-breakers)).

**Re-import with the current converter…** converts the same area again — the import's source file and box (or the
world's own copy of its area when the source has moved), the same centre (positions in the world do not move),
handedness, settings and map edits — into a **new** world named "… (re-imported)"; the old world and its runs stay.
The dialog shows the lane positions, lane links and traffic-light records of the current files (measured on the
spot for a world imported before the import measured them) beside the new conversion's, and adds the new world only
when asked. The traffic-light check is static (`core.xodr.signal_lanes`): a signal whose orientation names a side of
its road with no driving lane — the case CARLA gave no lane to control. A left-hand world imported by an earlier
version says so on the Worlds page. Central
Bengaluru as imported by 2.1.0, re-imported (1.0 s): lane positions 95th percentile 10.05 m → 0.58 m, misplaced ways
11 → 1 of 23; lane links over 0.5 m 202 → 150 of 1136 (95th percentile 2.25 m); signal records controlling no lane
23 → 0 of 23, and on the server the 7 lights then control 1–5 lanes each.

## Measured on central Bengaluru (CARLA 0.9.16, the reference machine)

1 km² around (12.9716° N, 77.5946° E) from `bangalore-roads.osm`, left-hand traffic, the Indian mixed-traffic profile:

| step | result |
|---|---|
| import (read, crop, convert, left-hand, check) | 0.7 s: 245 roads (19.0 km, one per travel direction), 95 junctions, 7 signals; driving lanes per direction 1: 162, 2: 42, 3: 2, 4: 2 |
| lane check against OSM | 22 of 23 comparable ways agree (95.7 %), a seeded sample of the 25 named, lane-tagged ways |
| world generation on the server | 2.1–2.6 s, 880 spawn points |
| a 20 s drive with 60 vehicles (23 motorcycles, 21 cars, 8 autorickshaw substitutes, 5 buses, 2 bicycles, 1 van) and 40 walkers | 46 steps/s with one 1280 × 720 camera and semantic labels |
| import to a recorded drive with traffic | 12–17 s (the acceptance asks for under 5 minutes) |
| handedness, measured | **left**: 100 % of 1044 samples of 40 vehicles keep left, 98.4 % move with their lane; 100 % of 3234 sampled map lanes run the left-hand way; the server removed 11 vehicles at dead ends of the cut area during the 40 s |

## Labels in an imported world (a CARLA limit, worked around)
Measured on CARLA 0.9.16 in central Bengaluru with a car parked 10 m ahead: the road and sidewalk mesh that
`generate_opendrive_world` builds is drawn by the RGB camera but **not by the label cameras** — the depth camera sees
nothing there (99.998 % of the lower third at its 1000 m far limit), the instance and semantic cameras tag it sky
(11) — while actors are seen normally (the car: 1346 instance pixels and 1340 depth pixels at a median 6.1 m). The
semantic lidar does hit the mesh, with tag 0.

Falcon labels the ground of such worlds from the map instead. These worlds are flat (OpenStreetMap has no elevation):
a pixel tagged sky whose ray meets the map's ground plane within 250 m gets that point's range and the class of the
map's lanes there, from a 0.25 m raster (driving lanes → road, sidewalks → sidewalk, other lane types → ground —
`Map.generate_waypoints` gives driving lanes only, so the sidewalks and shoulders are found beside them with
`Waypoint.get_right_lane`, and across the centre line of a one-way road; measured on the re-imported central Bengaluru: `generate_waypoints` gave 14 836 waypoints, all on driving lanes, though its file has 343 sidewalk lanes; found beside them, the sidewalks took 4.1 % of the pixels of a 10 s drive that had been labelled ground (road 29.1 %, ground 20.1 % → 16.0 %)); lidar
points tagged 0 within 0.3 m of the plane are named the same way. Each frame's `objects.json` (`ground_relabelled`)
and each run's lidar checks count them, and `dataset.json` states the method. In the Bengaluru drive this labelled
about 45 % of each frame (the rest sky); vehicles, walkers, poles and lights keep CARLA's own labels.

## The full-fidelity path (external, advanced)
For a place with its buildings, vegetation and textures, build a CARLA map package:

1. Get the road network as OpenDRIVE (Falcon's converted `.xodr`, or RoadRunner's export).
2. Model the scene in RoadRunner (roads from the `.xodr` or OSM, buildings from OSM footprints or photogrammetry,
   props, markings) and export **FBX + `.xodr`** for CARLA (RoadRunner's CARLA export).
3. In a **CARLA 0.9.16 source build with the Unreal editor**, import the package (`make import`), check the
   semantic tags of the meshes (they decide the labels), build the navigation mesh for pedestrians.
4. Cook it into a distributable package (`make package ARGS="--packages=<Name>"`) and install it into the CARLA
   folder Falcon uses; the map then appears in Tools ▸ Probe and on the Worlds page as a town.
5. Check: spawn points exist, traffic lights have trigger volumes, the traffic manager drives it, the semantic camera
   labels buildings and vegetation as such.
