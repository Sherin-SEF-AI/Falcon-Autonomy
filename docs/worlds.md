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
gives the agreement over the comparable ways and lists the others.

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
with right lanes only and its reference line at the street's centre.

**What Falcon does**: with *Left-hand traffic* on (the default for an Indian place), Falcon converts the converter's
output: every road's lanes are mirrored across its reference line (lane k → −k), every road gets `rule="LHT"`, lane
offsets, lane links, junction lane links and signal positions are mirrored with them. Each road keeps its travel
direction; only the side changes. After loading, a verification run spawns traffic and checks which side it drives
on; the report says so.

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
