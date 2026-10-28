# Worlds, places and traffic

## CARLA's towns
A town world names a CARLA map (Town01–05, Town10HD and their `_Opt` variants in the base package; Tools ▸ Probe lists
the server's own) with its weather and time of day. Every one of CARLA's towns is **right-hand traffic**.

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

**Autorickshaws** are not in CARLA 0.9.16. The profile substitutes `vehicle.micro.microlino` (a small three-door car,
the nearest size) and labels it *autorickshaw (substitute: Microlino)* everywhere it appears. To add a real one:
model it (or buy a model), import it into the CARLA Unreal editor build as a vehicle blueprint following CARLA's
"Add a new vehicle" guide (skeleton, wheels, physics), cook it into a package, and name its blueprint in the profile's
fleet (`blueprint:vehicle.<make>.<model>`). The traffic manager keeps vehicles in lanes, so lane-splitting by
two-wheelers is not reproduced by the profile.

## A standalone OpenDRIVE world: what it is
`generate_opendrive_world` builds **roads, lanes, sidewalks and junctions** as meshes from the OpenDRIVE file — no
buildings, trees, real textures, road paint beyond lane marks, or props. Falcon says so on the Worlds page.

CARLA can spawn at runtime only the props in its blueprint library (`static.prop.*`: fixed models such as cones,
barriers, benches, boxes); it cannot build arbitrary meshes such as extruded building footprints at runtime. Falcon
therefore does not place buildings in an imported world and reports the OSM extract's building count as *not
represented*.

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
