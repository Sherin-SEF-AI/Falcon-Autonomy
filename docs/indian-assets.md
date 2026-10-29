# Indian road assets (within the packaged CARLA)

CARLA 0.9.16's packaged release cannot import a new mesh: making one needs a build of Unreal Engine (a CARLA source
build, or CARLA's Docker image of Unreal to cook a package). Falcon builds what the packaged release does allow and
labels every approximation as a **stand-in** (a CARLA vehicle driving as an Indian vehicle class) or a **proxy** (a
prop of an Indian object's size). Everything below was measured on Falcon's own server (CARLA 0.9.16, the reference
machine, 2026-10-29) unless it says *cited* or *not verified*.

| Indian asset | what Falcon uses | label (objects JSON `subclass`) | verified on the server |
|---|---|---|---|
| autorickshaw | `vehicle.micro.microlino`, the nearest vehicle in measured size, with a three-wheeler's top speed (traffic manager) and acceleration (engine torque × 0.7) | `autorickshaw (stand-in)` | size, speed cap, acceleration, labels, a cut-in scenario in Bengaluru |
| cycle-rickshaw | `vehicle.bh.crossbike`, the nearest *bicycle* in measured size, top speed 15 km/h | `cycle-rickshaw (stand-in)` | size (scenarios only, not in traffic) |
| speed breaker | an `<elevationProfile>` hump in the OpenDRIVE world (IRC:99-1988: 3.7 m, 0.10 m) | — (the road surface) | the mesh's height, a vehicle's z, pitch and vertical acceleration |
| speed bump, table | elevation profiles (OSM `traffic_calming=bump`, `table`) | — | as above |
| pothole | **not expressible**; a full-width dip (0.08 m × 1.2 m) is the nearest | — | the mesh renders the dip; a lateral `<shape>` is ignored |
| waterlogging | CARLA's `precipitation_deposits` (global), per world and in a scenario variant | — | drawn on Town10HD, **not drawn on an imported world** |
| cattle | a cooked mesh (`SM_MapTable` × 0.78) of a non-descript cow's size, standing in the lane | `cattle (proxy obstacle)` | size (ray-cast), labels, the stray-cattle scenario in Bengaluru |
| handcart | `static.prop.haybale` (the nearest *unscaled* prop to a goods rickshaw's size) moved at walking pace, a pedestrian walking behind it | `handcart (proxy)` | moving a prop, labels, the handcart scenario in Bengaluru |
| street vendor | `static.prop.foodcart` (CARLA's European food cart) and a pedestrian | `street vendor (proxy)` | spawns and labels (no built-in scenario) |

`falcon assets list` prints the classes and their stand-ins; `falcon assets show CLASS` the sources and the ranked
candidates; `falcon assets inventory` re-measures the server; `falcon assets verify --out DIR` runs the checks below.

## What the packaged CARLA 0.9.16 can spawn (the inventory)

`falcon_autonomy/resources/carla/inventory-0.9.16.json` (made by `falcon assets inventory`: every blueprint spawned
alone on Town10HD_Opt in synchronous mode, its box read after two steps, then destroyed):

- **214 blueprints**: 41 vehicles, 52 walkers (adults, teenagers, children; 1.10–1.86 m tall), 98 `static.prop.*`,
  `static.prop.mesh`, `static.trigger.friction`, `util.actor.empty`, `controller.ai.walker` and 19 sensors. **No
  animal, three-wheeler, rickshaw or handcart.** Every vehicle, walker and prop spawned; their measured boxes, semantic
  tags and (vehicles) physics — mass, peak torque, maximum rpm, wheel radii, maximum steering — are in the file. The
  props' mesh paths are added from CARLA's `Default.Package.json`.
- The smallest cars: `vehicle.micro.microlino` 2.207 × 1.481 × 1.376 m (513 kg, 175 N·m, 4500 rpm), `audi.a2`
  3.705 m, `nissan.micra` 3.633 m. Two-wheelers 1.51–2.35 m long.
- `Actor.semantic_tags` gives cars `[14]`, two-wheelers `[4, 18]` / `[4, 19]`, walkers `[12]` — and **nothing for
  props**; the instance camera tags a prop by where its mesh lives: `Static/Static/…` meshes 20 (*static*),
  `Static/Dynamic/…` meshes 21 (*dynamic*).
- **`static.prop.mesh` works in the packaged build**: its attributes are `mesh_path`, `scale` (float, default 1.0) and
  `mass`. It spawned every cooked mesh tried by its Unreal path, including two the package does not list as props (a
  food-cart umbrella, `SM_Plane`). `scale` is uniform: Box01 × 2 was drawn 1.374 m high (a ray cast down; 0.692 m at
  × 1) — but the **reported bounding box keeps the unscaled centre** (extents scaled, centre z 0.340 instead of 0.680):
  Falcon writes a proxy's true box itself. A path that does not exist **spawns an empty actor without an error**
  (its box ±inf).
- Moving props (needed for a handcart): a `static.prop.*` moves with `set_transform` after `set_simulate_physics(False)`
  (a hay bale moved 1.5 m and 1 m, its pixels with it). A **`static.prop.mesh` does not move**: `set_transform` leaves
  it where it was *and resets its scale to 1* (Box01 × 2: 2021 instance pixels before, 519 after), `set_location`
  leaves it too (scale kept), and attached to a `util.actor.empty` it is not drawn at all (an empty actor also refuses
  `set_simulate_physics`). So a moving proxy is an unscaled prop; a scaled mesh is for still ones.

## Choosing a stand-in or a proxy

Size closeness is the root mean square of the relative differences of length, width and height from the real class
(`core.assets.size_score`). A prop may be turned 90° and, with a mesh path, scaled uniformly within 0.5–2 (the scale
minimising the score, in closed form); a moving proxy may not be scaled (above).

| class | real size (L × W × H) and source | chosen | its size | score |
|---|---|---|---|---|
| autorickshaw | 2.635 × 1.300 × 1.700 m — Bajaj RE, bajajauto.com specifications (*cited*) | `vehicle.micro.microlino` | 2.207 × 1.481 × 1.376 | 0.165 |
| cycle-rickshaw | 2.200 × 1.000 × 1.200 m — IRC:11-2015 Table 1, passenger rickshaw (*cited*) | `vehicle.bh.crossbike` | 1.509 × 0.866 × 1.612 | 0.280 |
| cattle | 1.61 × 0.44 × 0.85 m — from Thalkar et al. 2016 (*cited*, see below) | `SM_MapTable` × 0.777 | 1.222 × 0.479 × 0.917 | 0.155 |
| handcart | 2.400 × 1.220 × 1.200 m — IRC:11-2015 Table 1, goods rickshaw (*cited*) | `static.prop.haybale` | 1.248 × 1.193 × 1.195 | 0.277 |

- The cycle-rickshaw is chosen among CARLA's **bicycles** (a pedalling rider; semantic bicycle and rider). The nearest
  of all vehicles is `vehicle.yamaha.yzf` (score 0.177): a motorcyclist would mislabel it.
- Cattle: Thalkar MG, Bhagat DJ et al. (2016), *Morphological and morphometric features of non-descript cattle in
  Raigad district of Maharashtra*, Journal of Livestock Biodiversity 6(2): 360 animals, mean body length 95.69 cm,
  height at withers 85.10 cm, chest girth 139.55 cm, face length 39.96 cm. The length on the road (body + face +
  0.25 m of neck) and the width (girth / π) are **Falcon's estimates** (heuristic); the height is the measured one.
  Non-descript cattle are what stray in Indian cities; a breed animal is larger.
- The handcart: IRC:11-2015 names vendors' four-wheeled trolleys but gives no size; its goods rickshaw, the design
  vehicle for non-motorised goods, is used.
- The proxies **do not look like** what they stand for (a map table, a hay bale): a detector trained on them learns
  nothing about cows or carts. What they test is that something of that size and place is there (planning, braking,
  free-space and occupancy, lidar returns).

## Kinematic limits

| class | top speed | acceleration | where applied |
|---|---|---|---|
| autorickshaw | 65 km/h — a dealer listing of the Bajaj RE (trucks.tractorjunction.com; Bajaj's own page gives none) | 2.0 m/s² from rest — Pathivada & Vedagiri (2017), *Modeling acceleration characteristics of motorized three-wheeler*, ICAHETS 2017 (maximum 1.88–2.08 m/s², GPS data) | traffic manager and director |
| cycle-rickshaw | 15 km/h — IRC:11-2015 §5.2 (non-motorised traffic 5–15 km/h, rarely 20) | not found: the stand-in's own | director |
| handcart | 3–6 km/h, walking pace (Falcon's range, heuristic) | — | director |

**On the traffic manager** (`workers.scene.ClassLimits`): CARLA's `set_desired_speed` replaces the traffic manager's
limit-relative target, so Falcon applies the road's limit itself, re-read every second of simulation:
desired = min(limit × (1 − the vehicle's percentage difference), top speed). *Measured* on Town10HD (30 km/h roads):
a Microlino capped at 20 km/h ran at p95 19.3 km/h (max 19.5) while an uncapped one ran at p95 28.5 km/h, over 35 s.
On an imported world the limit is the lane's own from the OpenDRIVE file, with the look-ahead for lower limits and
speed breakers ([worlds: speed limits](worlds.md#speed-limits-and-speed-breakers)), re-read every 0.25 s.

**Acceleration** through CARLA's physics (the traffic manager drives through it): the stand-in's engine torque curve is
scaled. *Measured* at full throttle from rest on a flat generated road:

| torque scale | 0–30 km/h | peak acceleration over 0.5 s | top speed |
|---|---|---|---|
| 1.0 (CARLA's Microlino) | 2.90 s | 5.43 m/s² | 91 km/h |
| 0.82 | 3.56 s | 4.94 m/s² | 85 km/h |
| 0.76 | 3.86 s | 4.75 m/s² | 82 km/h |
| **0.7 (Falcon)** | **4.22 s (a mean 1.97 m/s²; 2.0 m/s² takes 4.17 s)** | 4.51 m/s² | 79 km/h |
| 0.6 | 5.02 s | 4.15 m/s² | 75 km/h |
| 0.4 | 8.32 s | 3.60 m/s² | 56 km/h |

The limit: the mean to 30 km/h matches, but CARLA's first-gear launch keeps a 0.5 s peak of 4.5 m/s² (a real
three-wheeler's peak is about 2 m/s²); the top speed is the traffic manager's cap, not the engine's.

**In scenarios** (kinematic motion, the default) the director ramps a stand-in's set speed no faster than its class's
acceleration and never above its top speed.

*Measured in central Bengaluru* (left-hand, the Indian profile, 60 vehicles, 15 s): 8 autorickshaw stand-ins, torque
× 0.7 applied, their speeds at most 36.7 km/h (the limit and the per-vehicle difference bound them before the cap);
592 labelled stand-in objects, all `class` car, `category` vehicle.car, `subclass` autorickshaw (stand-in).

## Labels

The semantic and instance class stays CARLA's: a stand-in autorickshaw is a car mesh, tagged car (14), nuScenes
category `vehicle.car`; the cattle proxy is a static mesh (20, `movable_object.pushable_pullable`); the hay-bale
handcart a dynamic one (21). Falcon adds `subclass` to the object rows of `objects/<camera>/<n>.json`, to
`actors.jsonl` rows, to run.json (`subclasses`: actor id → label) and to a scenario's report. A proxy's 3D box is its
true size (the scaled mesh's, not CARLA's report). [Dataset format](dataset-format.md#indian-stand-ins-and-proxies)
says what this means for training.

## Road features in OpenDRIVE (real geometry)

`core.roadfeatures` writes features into an imported world's elevation profiles; CARLA's `generate_opendrive_world`
builds the road mesh with them, and the vehicle physics drives over that mesh.

- **Profiles**: `irc_hump` — IRC:99-1988 §5.1, a rounded hump of 17 m radius, 3.7 m along the road, 0.10 m high,
  advisory crossing speed 25 km/h (written as the parabola through the same chord and height: radius 17.1 m at the
  top); `osm_bump` — OSM's bump, "about 30 cm or shorter" along the road, 0.075 m high (the height is Falcon's);
  `table` — 0.10 m with 2 m ramps (IRC's 1:20) and a 4 m top (the top is Falcon's); `dip` — a full-width 0.08 m ×
  1.2 m depression (Falcon's).
- **Placement**: from OSM's `traffic_calming` (bump → `osm_bump`, or an IRC hump if the user says so; hump and yes →
  `irc_hump`; table → `table`) on every non-junction road within 3 m (each travel direction's road), and/or an IRC hump
  every N m along every road longer than 60 m (IRC:99-1988 §5.2: 100–120 m apart) — the Worlds page's *Road
  features*, or the import (on by default). A feature must fit on its road with 0.5 m to spare (a road's elevation
  must be zero where it meets the next); `rumble_strip`, `cushion`, `chicane` and the rest are reported as not
  represented. `features.json` lists what was placed and skipped, with the OSM node.
- **Measured: what the mesh renders** — the surface height along the right lane's centre every 2 cm
  (`world.ground_projection`, a ray against the generated mesh) on a straight test road:

| vertex distance | IRC hump (0.10 m) peak / RMS error | OSM bump (0.075 m, 0.30 m) | table (0.10 m) | dip (−0.08 m) |
|---|---|---|---|---|
| 2.0 m (CARLA's default) | 0.0995 / 12 mm | **0.031 m** over a 4.5 cm-raised base: smeared over 4 m | 0.100 / 0 | −0.050, smeared |
| 0.5 m | 0.0999 / 2.5 mm | 0.0735 / 18 mm | 0.100 / 0 | −0.080 / 8 mm |
| **0.25 m (Falcon's)** | 0.0999 / 1.1 mm | 0.072 / 6.5 mm | 0.100 / 0 | −0.080 / 3.6 mm |
| 0.1 m | 0.100 / 0.3 mm | 0.0717 / 4 mm | 0.100 / 0 | −0.080 / 0.2 mm |

  A world with features is generated at 0.25 m: central Bengaluru (1 km², 159 features) took 2.88 s to generate
  (2.1–2.6 s at 2.0 m before). The map's waypoints carry the elevation (z 0.075 m a quarter of the hump from its
  centre, exactly the profile); the generated mesh's ray hits are labelled 0 (unlabelled), like its road.
- **Measured: what a vehicle feels** — driven by Falcon's speed and lane controller through CARLA's physics over the
  features (mesh 0.25 m, 50 Hz; on the flat road before each feature z and the IMU's vertical acceleration were
  constant and the pitch varied by at most 0.0005° s.d.):

| vehicle, speed | IRC hump: z rise / pitch range / IMU peak | OSM bump | table | dip |
|---|---|---|---|---|
| Microlino, 15 km/h | 9.1 cm / 8.3° / 2.0 m/s² | 1.9 cm / 2.1° / 11.7 m/s² | 10.5 cm / 6.7° / 1.4 m/s² | −4.3 cm / 5.7° / 4.9 m/s² |
| Microlino, 30 km/h | 9.5 cm / 6.5° / 5.1 m/s² | 1.2 cm / 1.2° / 19.7 m/s² | 11.3 cm / 5.9° / 3.1 m/s² | −3.4 cm / 3.1° / 10.7 m/s² |
| Lincoln MKZ 2020, 15 km/h | 5.3 cm / 4.5° / 1.8 m/s² | 1.9 cm / 1.6° / 11.2 m/s² | 10.5 cm / 4.4° / 1.4 m/s² | −3.8 cm / 3.1° / 4.7 m/s² |
| Lincoln MKZ 2020, 30 km/h | 5.2 cm / 4.4° / 4.2 m/s² | 1.3 cm / 1.1° / 19.6 m/s² | 10.8 cm / 4.5° / 3.0 m/s² | −3.1 cm / 2.6° / 10.2 m/s² |

  (z of the vehicle's origin: a long wheelbase averages a short hump; the short bump is a jolt, not a lift). At 2.0 m
  the bump was felt as a 5.5 cm, 4 m hump instead. **The kinematic director does not feel them**: scenario actors in
  kinematic motion (the default) keep the height of their start; `"motion": "physics"` drives over them.
- **Potholes**: a pothole is local across the lane; OpenDRIVE says that only with `lateralProfile <shape>`. *Measured*:
  CARLA 0.9.16's generator **ignores `<shape>`** (a shape raising the right side 0.10 m per metre left both lanes at
  0.000 m; the sidewalks stand at +0.152 m as always). A full-width dip is the only depression it renders; Falcon
  offers it as `dip` and does not call it a pothole.
- **Traffic rules**: the Indian rule set's `speed_breaker` rule (heuristic: not above the breaker's advisory 25 km/h
  while the ego's body is over it; signals `on_breaker`, `breaker_kmh` from the world's file). On the 15 s Bengaluru
  drive the ego never crossed one (satisfied, vacuously); the rule's logic is checked on synthetic traces in the tests.
- **Central Bengaluru's tags** (1 km² around 12.9716° N, 77.5946° E): one `bump` node, placed on both travel
  directions' roads; one `hump` and three `table` nodes skipped (in junctions or at a road's end); five
  `rumble_strip` not represented. The whole extract has 1 bump, 1 hump, 12 tables, 12 rumble strips.

## Waterlogging

CARLA's puddles are `precipitation_deposits`, one value for the whole world (`World.set_weather` has no place): there
is no puddle prop or decal blueprint in the library, and `static.prop.mesh` spawns meshes, not decals, so **local
puddles are not possible** in the packaged build. *Measured* with a 640 × 360 camera, deposits 0 against 100 (no rain,
no wetness): on Town10HD the image's lower half changed by a mean 26.3 levels (82.5 % of its pixels by more than 10);
on a generated OpenDRIVE road by 0.27 (0.02 %) — **puddles are not drawn on an imported world**. Falcon exposes
deposits per world (Worlds ▸ weather, *Puddles*) and in the scenario variant **Monsoon: waterlogged road** (the lead
vehicle brakes in heavy rain with deposits 100), which says so when run on an imported world. *Not verified*: whether
CARLA's wet road or puddles change tyre grip (Falcon assumes not and states it).

## Scenarios

**Stray cattle in the lane** (`stray_cattle`), **Handcart in the kerb lane** (`handcart_kerb_lane`: the cart at
3–6 km/h near the kerb, its pusher walking behind), **Autorickshaw cuts in** (`autorickshaw_cut_in`) and **Monsoon:
waterlogged road** (`monsoon_waterlogged`), beside the earlier Indian four. *Measured* in central Bengaluru
(left-hand), 320 × 180 labelled camera: stray cattle passed (minimum time to collision 2.16 s; 280 labelled proxy
objects, class static); the handcart moved 15.9 m in 16 s (3.58 km/h) with its pusher alongside, passed (320 labelled
objects, class dynamic); the autorickshaw cut in (seeds 3 and 5: at 1.30 s and 0.35 s, at most 40.8 and 47.3 km/h),
passed (227 and 240 labelled objects, class car); the monsoon variant passed on Town10HD with deposits 100.

## Limits

- Stand-ins and proxies have the wrong appearance; only their size, motion and place are Indian.
- No three-wheeled physics: the autorickshaw stand-in is a four-wheeled car (its turning and roll are a car's).
- Cycle-rickshaws are not in the traffic profile (the traffic manager's bicycle pool cannot tell a rickshaw stand-in
  from a bicycle); they appear in scenarios.
- Speed breakers span the full road width; potholes are not representable; puddles are global and not drawn on
  imported worlds.
- The OpenSCENARIO export writes a proxy as a Vehicle (category car) with `falcon:kind` saying what it is.
- *Not verified*: the cycle-rickshaw stand-in on the server (sized from the inventory only), a real three-wheeler's
  acceleration curve beyond its maximum, tyre grip on wet roads.

## The full path: real meshes with a CARLA source build (not done here)

Read from CARLA 0.9.16's documentation (`build_linux`, `tuto_A_add_vehicle`, `tuto_A_add_props`) and the packaged
install; **not done or run in Falcon**.

1. **Build CARLA from source** on Ubuntu 20.04 or 22.04: about 130 GB of disk (CARLA ~31 GB, Unreal ~91 GB), an
   NVIDIA RTX 2000-series GPU or better with ≥ 6 GB (better 8 GB), a 4-core i7 or equivalent, Python ≥ 3.8; several
   hours. Unreal is CARLA's fork of **UE 4.26**: `git clone --depth 1 -b carla https://github.com/CarlaUnreal/UnrealEngine.git`,
   then `./Setup.sh && ./GenerateProjectFiles.sh && make` (no `-j`; 1–2 h). CARLA: `git clone -b ue4-dev
   https://github.com/carla-simulator/carla`, `./Update.sh` (the content), `make PythonAPI`, `make launch` (the
   editor). Pin the release tag matching 0.9.16 so the server and the 0.9.16 client agree.
2. **A vehicle (an autorickshaw)**: CARLA documents 4-wheeled, N-wheeled and 2-wheeled vehicles, **not 3-wheeled** —
   a three-wheeler needs the N-wheeled base (`BaseVehiclePawnNW`) with its own wheel setup, or a 4-wheel rig with the
   front wheels close together. Model at real scale (50 000–100 000 triangles; LODs), bind CARLA's vehicle skeleton
   (bones `Wheel_Front_Left`, `Wheel_Front_Right`, `Wheel_Rear_Left`, `Wheel_Rear_Right`, `VehicleBase`), split the
   materials (Bodywork, Glass_Ext, …), export the body, the physics collision mesh `SMC_<name>` (`UCX_` parts) and
   the ray-cast mesh `SM_sc_<name>`. In the editor: import into `Content/Carla/Static/Vehicles/4Wheeled/<name>`, set up
   the physics asset (wheel spheres kinematic), the animation blueprint (`VehicleAnimInstance`), `BP_<name>` from
   `BaseVehiclePawn` with wheel blueprints (`Wheel_Shape`, steer angle 70 front, handbrake rear), then register it in
   `VehicleFactory` (Make, Model, Class). Test with `manual_control.py --filter <model>`.
3. **Props (a cow, a cart, a vendor's thela)**: an import package — `Import/<Package>/<Package>.json` with
   `{"props": [{"name": <as the .fbx>, "source": "./Props/<Name>/<Name>.fbx", "size": tiny|small|medium|big|huge,
   "tag": <a semantic tag: Static, Dynamic, …; misspelled → Unlabeled>}], "maps": []}` beside `Props/<Name>/<Name>.fbx`
   (and textures). In the source build, `make import` puts them into Unreal's content and writes
   `<Package>.Package.json` into the package's `Config`, which defines them in the blueprint library. **There is no
   animal tag**: a cow would be Dynamic or Other unless the source is changed. For the *packaged* CARLA, the props are
   cooked into `<Package>.tar.gz` with CARLA's Docker image of Unreal (`Util/Docker/docker_tools.py --input … --output
   … --packages <Package>`; building the image takes about 4 hours and 400 GB the first time), then extracted with the
   package's own `ImportAssets.sh` from its `Import/` folder (present, empty, in the reference install).
4. **Package** with `make package` and install into the CARLA
   folder Falcon uses; run `falcon assets inventory` again: the new blueprints then appear in the inventory and
   `core.assets` can name them (`fleet: blueprint:vehicle.<make>.<model>` in a traffic profile; drop the stand-in).
