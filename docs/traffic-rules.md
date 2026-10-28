# Traffic rules

Falcon checks drives against traffic rules written as **signal temporal logic** (STL) formulas over what a run
recorded: the ego's pose and speed, the other actors, the map's lanes, junctions, speed limits and crosswalks, and the
traffic-light state where it is recorded. Each rule gets a **robustness** value (how far the run was from breaking it,
or how badly it broke it), the **intervals** in which it was violated with their times, and **counts per km**. Code:
`falcon_autonomy/core/stl.py` (the formula language and its evaluator) and `falcon_autonomy/core/rules.py` (signals,
rule sets, checking); their docstrings are the reference.

It runs on recorded drives, stack runs, live recordings and replays (any run folder with `frames.jsonl`), on gym and
Leaderboard episodes (from the per-step trace the environment now logs), in stack suites, in the safety-case report and
in an HTML report of its own. No dependency was added: the evaluator is Falcon's own.

## Where it shows

| where | what |
|---|---|
| `falcon rules check RUN... [--set india\|generic\|FILE] [--xodr FILE\|none] [--out DIR] [--json]` | each run's rules (and an `episodes.jsonl` of gym or Leaderboard episodes); `--out` writes `rules.json` with each set's aggregate over the runs |
| `falcon rules list` · `show SET` · `signals` | the built-in sets and their rules; a set's file (to copy and edit); the signals a rule can read |
| Data ▸ Browse ▸ *Traffic rules* | the run's check in the background, a timeline under the frames (red: violated; grey: not recorded), each rule's state, robustness, violations, per km and the share of time evaluated; click the timeline or double-click a rule to go to that frame; *Save for a report* writes `workspace/rules/<run>-<set>/rules.json` |
| Training ▸ Leaderboard | *Traffic rules*: the routes' violations and per km beside the driving score (not in it), the rules violated most; a *Rule violations* column per route |
| `leaderboard.json` | `rules` (the aggregate) and per route `rule_violations`, `rule_violations_per_km` |
| stack suites (`results.json`) | `rules`: the aggregate and per scenario; each run's check in its folder as `rules.json`; a suite may name its set (`"rules": "india"`, `"generic"` or a file) |
| safety-case report | a *Traffic rules* section (per rule: runs evaluated, violations, per km with its interval, runs violating with Wilson's interval, the lowest robustness) and a residual risk per rule violated |
| `falcon report rules DIR --out FILE` (Reports ▸ *Traffic-rule check*) | the set and its formulas, each run's timeline and table, the aggregate with intervals; the Leaderboard report gains the rules section too |

Violations are results, not failures: `falcon rules check` exits 0 when it could check, 2 when it could not (no
`frames.jsonl`, a rule file that does not parse).

## The formula language

```
φ -> ψ        φ or ψ        φ and ψ        not φ        ( φ )        true   false
φ until[a,b] ψ                φ since[a,b] ψ
always[a,b] φ   (G)           eventually[a,b] φ   (F)       — over [t + a, t + b]
historically[a,b] φ   (H)     once[a,b] φ   (O)             — over [t − b, t − a]
e1 <= e2   e1 < e2   e1 >= e2   e1 > e2                     a predicate
NAME                                                        a boolean signal: NAME >= 0.5
expressions: numbers, signal and constant names, + − * /, unary −, abs(e), min(e, …), max(e, …), default(e, d)
```

Precedence, loosest first: `->` (right-associative), `or`, `and`, `until`/`since`, then `not` and the temporal prefixes.
Intervals are in seconds (constants allowed, `inf` for unbounded; `G φ` is `G[0, inf] φ`), with `0 ≤ a ≤ b`. The parser
is a recursive-descent parser over a small token set: nothing is evaluated as Python, and an error names its column.

**Quantitative semantics** (Fainekos & Pappas 2009; Donzé & Maler 2010): ρ(e1 ≥ e2) = e1 − e2, ρ(e1 ≤ e2) = e2 − e1
(`>` and `<` have the same robustness: strictness is not represented, and a robustness of exactly 0 counts as
satisfied); `not` negates; `and` is the minimum, `or` the maximum, `φ -> ψ` is max(−ρφ, ρψ); `always`/`historically`
are the minimum over the window, `eventually`/`once` the maximum; `until` is max over t′ of min(ρψ(t′), min over
[t, t′) of ρφ), and `since` its mirror into the past. ρ ≥ 0 is satisfied. Its magnitude is in the predicate's units
(metres, m/s, km/h, degrees); a conjunction of predicates in different units compares different units — the usual
STL caveat, and why each rule reports its own robustness.

**Finite traces.** Windows are cut at the trace's ends: `always` over an empty window is +∞, `eventually` −∞. A
look-ahead near the end of a run checks only what remains, and a look-back near its start only what came before —
so rules that look back for "longer than X s" also require `t ≥ X` (below).

**Unknowns (three-valued).** A signal is NaN where it is unknown: not recorded, or not measurable there. `and` with an
unknown is the known side when that side is negative (violated whatever the unknown is), else unknown; `or` with an
unknown is the known side when it is ≥ 0; a window with an unknown sample is unknown unless its known samples decide
it. **An unknown never makes a violation**: it leaves the sample unknown, and each rule reports the share of time it
could be evaluated. `default(e, d)` is how a rule supplies a value.

**A rule** is a formula that must hold at every sample. The run's robustness for it is ρ(G φ), the minimum over the
known samples; a violation is a maximal interval of samples with negative robustness (its start, its end — the last
violating sample plus one step — and its most negative value and when); counts per km divide the violations by the
distance the ego drove (from its positions). A rule unknown throughout is *not evaluable* and names the signals that
were missing.

### Checked by hand

`tests/test_rules.py` checks every operator against robustness worked out by hand on synthetic signals: the predicates
and arithmetic, `and`/`or`/`->`, bounded and unbounded `always`, `eventually`, `historically`, `once` (including empty
windows), `until` and `since`, windows in seconds on non-uniform sampling, the three-valued unknowns, the violation
intervals, the parser's precedence and errors.

## Rule files

```
# a comment
ruleset india "India (left-hand traffic)"        the set's id and name
handedness LHT                                   the traffic side it is written for
about "…"                                        the set's description (repeated lines continue it)
let follow_time_s = 2   # heuristic: …           a constant (numbers only); the comment is kept as its note
rule following_distance "Keep a safe distance from the vehicle ahead" heuristic:
    speed_mps < 1 or lead_gap_m >= follow_time_s * speed_mps
about following_distance "…"                     the rule's explanation (repeated lines continue it)
```

A formula may continue on indented lines. Tags: `law` (it restates a traffic law) and `heuristic` (a threshold or a
reading of the law that is Falcon's own choice). Errors name the line: an unknown signal or statement, a constant that
shadows a signal, a rule defined twice. `falcon rules show india > my.rules` writes a set to edit; `--set my.rules`
(or a suite's `"rules"`, or Data ▸ Browse ▸ *A rule file…*) uses it. The set used is recorded with its SHA-256 in
every result.

## The signals

| signal | unit | what |
|---|---|---|
| `t` | s | a run's time from its first sample; an episode's simulation time |
| `speed_mps`, `speed_kmh`, `accel_mps2` | | the ego's speed (frames.jsonl; when it disagrees with the positions — kinematic scenario drives recorded before 0.6 logged 0 — the positions' central difference, said in the result) and its derivative |
| `on_lane` | 0/1 | the ego's centre is on a driving lane |
| `in_junction` | 0/1 | the lane is a junction road in the map's OpenDRIVE; without the file a heuristic (below); gym traces: CARLA's `Waypoint.is_junction` |
| `lane_heading_err_deg` | ° | \|ego yaw − its lane's heading\| (where lanes overlap, the lane nearest its heading) |
| `lanes_left`, `lanes_right` | lanes | same-direction driving lanes beside the ego's lane (heading within 30°, measured across the lane's cross-section) |
| `lane_margin_m` | m | lane width/2 − \|lateral offset\| − half the ego's width: negative when the body is over a line |
| `solid_margin_m` | m | the body's distance to a `Solid` or `SolidSolid` line of its lane (CARLA's markings in lanes.json); +∞ when neither line is solid; unknown without markings |
| `lead_gap_m` | m | bumper to bumper to the nearest vehicle ahead in a straight corridor in the ego's frame (\|lateral\| < the half widths + 0.3 m, heading within 60°, up to 80 m); +∞ when none |
| `ped_path_gap_m` | m | to the nearest pedestrian in the path (\|lateral\| < the half widths + 1 m, up to 50 m); +∞ when none |
| `ped_on_crossing` | 0/1 | that pedestrian stands on a crosswalk of the map (OpenDRIVE crosswalk objects); unknown when the map's crosswalks are not known |
| `pass_on_left`, `pass_on_right`, `passing` | 0/1 | at the sample where the ego passes a same-direction vehicle (its centre goes from ahead to behind the ego's while 1.4–8 m beside it, the ego faster) — with the ego on that vehicle's left or right |
| `tl_red`, `tl_distance_m` | 0/1, m | the light on the route ahead is red; front to its stop line (gym traces; drives do not record light states) |
| `speed_limit_kmh` | km/h | CARLA's `Vehicle.get_speed_limit()` where a trace records it; else the map's OpenDRIVE record for the ego's road (a lane `<speed>`, else the road type's); in a junction road the last limit before it |

The corridors are straight lines in the ego's frame: on a curve they are a heuristic.

### The map

- **lanes.json** (drives that record labels): CARLA's driving-lane waypoints with widths, road/section/lane ids and
  CARLA's lane markings.
- **The OpenDRIVE file**: the drive's own (an imported world), or for a CARLA town
  `CarlaUE4/Content/Carla/Maps/OpenDrive/<town>.xodr` under the CARLA folder in Settings (or the usual places), or
  `--xodr`. It gives the junction roads, the speed records and the crosswalks; and the lane points themselves when the
  run has no lanes.json (each driving lane's centre every 2 m from the reference line, laneOffset and widths; OpenDRIVE
  (x, y, heading) → CARLA (x, −y, −heading); the header's offset not applied; markings unknown).
- **A town's file is checked against the run's lanes.json before it is used**: at least 90 % of lanes.json's points
  must have a lane centre made from the file within 0.25 m laterally and the same heading; otherwise the file is not
  used and the result says so (another map or CARLA version).
- **Junctions without OpenDRIVE** (heuristic): a road is a junction road when its lane points overlap those of two or
  more other roads at headings 20°–160° apart.
- **Speed records in junction roads are ignored**: SUMO's converter writes curvature speeds there, not limits.

## The rule sets

`india` — **India (left-hand traffic)**. The Motor Vehicles (Driving) Regulations, 2017 as the rules restate them.

| rule | formula | tags |
|---|---|---|
| `speed_limit` | `speed_kmh <= default(speed_limit_kmh, urban_limit_kmh) + speed_tolerance_kmh` | law, heuristic |
| `stop_at_red` | `not (tl_red and tl_distance_m <= red_line_m) or speed_mps <= 0.5` | law |
| `no_wrong_way` | `in_junction or not on_lane or lane_heading_err_deg <= 135` | law |
| `keep_left` | `in_junction or not on_lane or lanes_left < 0.5 or once[0,overtake_window_s] passing or eventually[0,overtake_window_s] passing` | law, heuristic |
| `overtake_right` | `not pass_on_left` | law |
| `lane_discipline` | `in_junction or not on_lane or t < lane_change_s or once[0,lane_change_s] (lane_margin_m >= 0)` | law, heuristic |
| `solid_line` | `in_junction or solid_margin_m >= 0` | law |
| `following_distance` | `speed_mps < 1 or lead_gap_m >= follow_time_s * speed_mps` | heuristic |
| `yield_crossing` | `not ped_on_crossing or speed_mps <= 0.5 or ped_path_gap_m - speed_mps * speed_mps / (2 * comfort_decel_mps2) >= stop_margin_m` | law |
| `pedestrian_clearance` | the same without `not ped_on_crossing`: anywhere (on Indian roads pedestrians cross away from crossings) | heuristic |
| `junction_clear` | `not in_junction or t < junction_stand_s or once[0,junction_stand_s] (speed_mps >= 0.5) or lead_gap_m <= queue_gap_m` | law, heuristic |

`generic` — **Generic (right-hand traffic)**, not any one country's law: the same rules with `keep_right`
(`lanes_right`) and `overtake_left` (`not pass_on_right`), without `pedestrian_clearance`.

**Heuristic constants** (Falcon's choices, in each set's file to edit): an urban limit of 50 km/h where the map records
none (Indian limits are notified per state and city); no speed tolerance; the two-second rule (the regulations ask for
a distance sufficient to stop, without a number); a firm stop at 4 m/s² with 2 m to spare before a pedestrian; the stop
line reached within 1.5 m (one step at 108 km/h and 20 Hz); a lane change within 3 s; out of the kerb lane only within
10 s of passing a vehicle; standing more than 5 s in a junction blocks it unless queued within 10 m of a vehicle.

The default set is chosen by the run's handedness (LHT: india; else generic). A set written for the other side is
allowed but warned about: its keep-side and overtaking-side rules then read the wrong side.

### What is not recorded, so not checked

- **Turn indicators.** CARLA's recorded state has none: lane changes and turns without signalling are not checked, and
  the overtaking exceptions that depend on a signal (passing on the left a vehicle turning right and signalling) are
  counted as passes on the left.
- **Traffic-light states in drives.** Drives do not record them, so `stop_at_red` is evaluated only on gym and
  Leaderboard traces (and on runs whose frames carry a `traffic_light` block).
- **Right of way at unsignalled junctions** (OpenDRIVE junction priorities are not read), horn use, vehicle condition.
- **Legal speed limits** where the map has none (the set's default, labelled heuristic). On imported OSM worlds the
  map's lane speeds are SUMO's (OSM `maxspeed`, else SUMO's road-type default) — a stand-in for the notified limit.
- **Crosswalks** only where the OpenDRIVE file has `crosswalk` objects (CARLA's towns do; OSM imports have none, so
  `yield_crossing` is unknown there and `pedestrian_clearance` applies).
- **Gym traces** carry the lead and the vulnerable-road-user gaps of the environment's observation (a cyclist counts as
  a pedestrian there), not other actors' poses: no passing events, no crosswalks.

## Measured

On the reference machine, 2026-10-29.

**Real CARLA 0.9.16 runs** (recorded earlier; no server was run for this):

| run | samples | km | set | result |
|---|---|---|---|---|
| Town10HD_Opt autopilot drive, 40 vehicles, 20 walkers (`m5/drive`) | 160 | 0.057 | generic | `keep_right` violated 0–8 s (the traffic manager drove the inner lane of two; no pass): 1 violation, 17.6 per km; all other rules satisfied; `speed_limit` robustness 32.6 km/h against Town10's 40 mph (64.4 km/h) road record; `stop_at_red` unknown (light states not recorded) |
| Town10HD_Opt `lead_brake` scenario (`m5/lead`) | 240 | 0.081 | generic | `keep_right` violated twice (0–2.2 s, 6.6–12 s; a junction between); `following_distance` satisfied with 1.64 m to spare at the tightest (the follower stopped 5.2 m behind the braking lead); its recorded speed was 0 (recorded before 0.6): the positions' speed was used, and said so |
| imported Bengaluru, left-hand, no actors.jsonl, no lanes.json (`gt/blr`) | 120 | 0.039 | india | no violation; lanes, junctions and the 50 km/h limits made from the world's OpenDRIVE; `overtake_right` not evaluable (no actors), `solid_line` evaluated 62 % (markings unknown from OpenDRIVE; the rest in junctions) |

Checking one of these runs took 0.19–0.37 s (with the OpenDRIVE file parsed and checked against lanes.json); a
synthetic 12 000-sample trace (10 minutes at 20 Hz) took 0.42 s for the India set.

**The OpenDRIVE reading against CARLA's own lanes** (Town10HD_Opt.xodr against the lanes.json CARLA wrote for
`m5/lead`): lane centres made from the file lie a median 0.4 mm (95th percentile 3 cm, worst 1.0 m) across from
CARLA's waypoints, headings agree (median 0°, 95th percentile 4.1°), 87 % carry the same road and lane id at the
nearest point; the agreement check passes at 100 %. The 16 crosswalk polygons (2.5 × 13–26 m) have their centres 1.0–3.9 m
from a lane point, 11 of 16 inside a lane's stretch — consistent with crossings laid across the roads; not checked
against the markings CARLA renders. The junction heuristic agrees with the
file's junction roads at 87 % of lane points (it marks 35 % of points as junction, the file 48 %).

**The gym trace's cost**: on the fake carla, 3075–3306 steps/s with the trace and 3152–3594 without (two runs each of
327 steps; within the run-to-run spread). Not measured on a real server.

## Limits

- The checker reads the recorded state at the recorded rate: an event between samples (a red light crossed at high
  speed between two steps) is seen only through the samples around it — hence the 1.5 m stop-line margin.
- Corridors (lead, pedestrian, passing) are straight in the ego's frame; on curves they are heuristic. Boxes are the
  recorded extents; the ego's is assumed (CARLA's mkz_2020, 4.9 × 1.84 m) when a run does not record it.
- Robustness mixes units across predicates of one rule (the usual STL caveat); compare a rule's robustness across runs,
  not across rules.
- The violation counts per km are counts of intervals: one long violation is one, however long; `violated_s` gives
  its duration.
- The India set restates the regulations for what is recorded; it is not a legal opinion, and its heuristic thresholds
  are Falcon's. The generic set is not any country's law.
- `lanes_left`/`lanes_right` in gym traces come from CARLA's `Waypoint.get_left_lane`/`get_right_lane` (direction-
  aware by CARLA's documentation); on left-hand OpenDRIVE roads that is not verified on a server.
- CARLA's own speed limit (`Vehicle.get_speed_limit`) is recorded by gym traces where the API has it; whether CARLA
  updates it for a kinematic (physics-off) ego was not verified, and on Town10 it may differ from the OpenDRIVE road
  record (CARLA reads speed-limit signals; not verified here).
