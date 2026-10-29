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
drawn on an imported world ([Indian road assets](indian-assets.md)). The traffic manager keeps vehicles in lanes, so lane-splitting by
two-wheelers is not reproduced by the profile.

## A standalone OpenDRIVE world: what it is
`generate_opendrive_world` builds **roads, lanes, sidewalks and junctions** as meshes from the OpenDRIVE file — no
buildings, trees, real textures, road paint beyond lane marks, or props. Falcon says so on the Worlds page.

CARLA can spawn at runtime only the props in its blueprint library (`static.prop.*`: fixed models such as cones,
barriers, benches, boxes); it cannot build arbitrary meshes such as extruded building footprints at runtime. Falcon
therefore does not place buildings in an imported world and reports the OSM extract's building count as *not
represented*.

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
Traffic vehicles that leave a dead end are not put back: they fall and are removed. CARLA's own towns are closed and
never trigger the guard.

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
| speed limit | written as the road's `<type><speed max>` (20 km/h → 5.56 m/s) | **not used**: the traffic manager drives to the vehicle's limit, which stays 30 km/h (a car drove at 28.9 km/h on a road edited to 20). A speed-limit signal (OpenDRIVE type 274) becomes a map landmark, but a generated world spawns no speed-limit sign, so the limit does not change either (tried at 20 and 40 km/h) |
| traffic signals | a traffic-light signal per approach and a controller (one junction: 4 signals, 8 controllers) | 4 traffic-light actors (green and red seen) that control their lanes with this version's left-hand fix; a car stops at red |
| speed breaker | — (placed by Falcon as an elevation profile) | driven over: the car rose 4–5 cm crossing an IRC hump at 29 km/h (mesh at 0.25 m) |

So a speed limit edited on the map is recorded and drawn but does not slow traffic in CARLA; set the traffic
profile's *Speed below the limit* (or a scenario's speeds) for slower traffic.

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
map's lanes there, from a 0.25 m raster (driving lanes → road, sidewalks → sidewalk, other lane types → ground); lidar
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
