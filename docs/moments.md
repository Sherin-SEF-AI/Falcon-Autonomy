# Moments: search every run for what happened

`falcon moments` indexes every recorded run of a workspace by the **events** in it — a cut-in, hard braking, a
pedestrian crossing, a near-miss — and finds them again from plain words:

```
falcon moments index  my.falcon.json
falcon moments search my.falcon.json "two-wheeler cuts in from the left at night in rain"
understood as: type cut_in · actor motorcycle or bicycle · side left · light night · weather rain
a3f09c1e-0   cut_in              motorcycle     left  TTC 4.20 s  sev 0.30     2.45–5.80   runs/20261120-181203
7be0d412-1   cut_in              motorcycle     right TTC 6.10 s  sev 0.30     3.10–6.40   runs/20261118-093011   [partial: not side, tod, weather]
2 moment(s), 1 matching every term
```

Everything runs on this machine: no model, no embedding, no network. The run search of the Data page
([runs](runs.md)) finds runs by what was *configured* (world, weather, tags); this finds moments by what *happened*.

## What is read

| Source | Files | Notes |
|---|---|---|
| a run folder (drives, scenario runs, a study's runs, replays of imported logs, live recordings) | `frames.jsonl`, `actors.jsonl` (or `trajectories.json`), `lanes.json`, `run.json`, the map's OpenDRIVE file when the run names one | through the traffic-rule checker's trace ([traffic rules](traffic-rules.md)) |
| an imported log | `logs/<name>/log.json` | sampled at 10 Hz; velocities from the positions |
| a near-miss ranking | `ranking.json` of `falcon dashcam ingest` ([near-misses](nearmiss.md)), given once with `--nearmiss DIR` | one moment per mined event; **private** |

Nothing is written into a run folder. The store is `<workspace>/moments/moments.sqlite` and can be deleted at any
time. It is SQLite (Python's standard library) rather than Parquet because the index is incremental — one run's rows
are replaced in one transaction — and because pyarrow is optional in Falcon.

`falcon moments index` reads only what is new or changed: a source is keyed, as the run index keys its cache, by its
path and the sizes and modification times of the files read, plus the extractor's version. `--run DIR` indexes one
run, `--force` everything again, `--no-map` skips the search for the map's OpenDRIVE file. With the setting
`moments_auto_index` (off by default, in `settings.json`) a small CPU job indexes each run when it finishes.

## Moment types

Geometry is in the ego's frame at each recorded step, with box-to-box distance, time to collision (TTC) and required
deceleration from the criticality measures of [scenarios](scenarios.md) (exact for oriented boxes). "In the ego's path" is a straight corridor: half the
two widths plus 0.3 m.

| Type | Found when | Key numbers |
|---|---|---|
| `cut_in` | a same-direction vehicle moves from beside the corridor into it, ahead, within 25 m, and stays 0.5 s | side, gap at entry, relative speed, min TTC over the next 3 s |
| `cut_out` | a lead within 40 m leaves the corridor to the side | side, gap |
| `hard_brake_ego` | the ego decelerates at 3 m/s² or more for 0.25 s | peak deceleration; the actor ahead, when there is one |
| `hard_brake_lead` | a vehicle ahead in the corridor does | peak deceleration, gap, min TTC |
| `close_pass` | an actor alongside is closer than 1.0 m laterally with 1 m/s or more speed difference | side, lateral clearance |
| `overtake` | a moving same-direction vehicle goes from ahead to behind (the ego passes) or behind to ahead, within 5.25 m laterally and 15 s | side, `by` (ego or actor) |
| `crossing` | a pedestrian or animal moves across the corridor ahead (0.3 m/s or more, within 40 m) | the side it came from, gap, walking speed |
| `jaywalking` | a crossing pedestrian is off every crosswalk the map knows; when the map's crosswalks are not known, the crossing is not at a junction (a heuristic, said in `numbers.basis`) | as crossing |
| `oncoming_in_lane` | a moving vehicle heading against the ego (135° or more) is in the corridor for 0.5 s | gap, closing speed |
| `stopped_obstacle` | anything stands (under 0.3 m/s for 1 s) in the corridor ahead while the ego moves | gap |
| `red_light_violation` | the ego's front is past the stop line of a red light, moving | only runs that record the light (gym traces) |
| `low_ttc` | TTC under 1.5 s without contact | min TTC |
| `high_decel_needed` | required deceleration at or over 3.35 m/s² | the deceleration |
| `near_miss` | no contact, and TTC under 1.0 s or the boxes within 0.5 m at 2 m/s or more | distance, min TTC |
| `collision` | the boxes overlap | relative speed at contact |
| `lane_departure` | the ego's centre is off every driving lane of the map for 0.25 s | needs `lanes.json` or OpenDRIVE |
| `sensor_fault` | `run.json` records a fault applied (over its `onset_s`–`until_s`, else the whole run) | the fault and its sensor |
| `mrm` | `run.json` holds a verified minimal-risk manoeuvre | reaction time, whether it passed |

Each moment also carries its actor classes — Falcon's own, the Indian kinds among them (`motorcycle`, `bicycle`,
`autorickshaw`, `cycle_rickshaw`, `cattle`, `handcart`, `vendor` …) from the actor's recorded subclass — its time
span, and its context: time of day and weather (from the run's configuration, as the run index names them), the
world, the road (`junction`, `speed_breaker` from the map's road features, else `road`) and the ego's speed band.

Every threshold is a named constant in `falcon_autonomy/core/moments.py` with where it comes from; most are Falcon's
own heuristics or shared with the scenario checker. Two are thresholds commonly cited in the traffic-conflict
literature — 1.5 s for TTC and 3.35 m/s² for required deceleration — **recalled, not checked against the documents**.
Changing one bumps the extractor's version, and every run is read again.

**Severity** (0–1) orders the hits: 1 for a collision; otherwise the largest of the type's base value,
0.9·(1 − min TTC / 3 s) and 0.9·(deceleration / 8 m/s²). It is a ranking heuristic, not a probability of harm.

## Queries

Plain words are turned into a filter by a fixed vocabulary — the scenario checker's phrase set
([scenario from text](scenario-text.md)) plus phrases for the moment types above:

- **actors**: car, truck, lorry, bus, van, tempo, auto / autorickshaw / tuk tuk, bike / scooter / scooty (a motorcycle,
  as in Indian English), cycle / cyclist, two wheeler (motorcycle or bicycle), pedestrian, child, cow / cattle /
  buffalo, handcart, vendor, animal, VRU;
- **what happened**: cuts in, cuts out, brakes hard, ego brakes, lead brakes, close pass, overtakes, crosses,
  jaywalking, oncoming / wrong way, stopped / standing / blocking, ran a red light, low TTC, near miss, collision,
  lane departure, sensor fault, MRM;
- **where and when**: from the left / right, at night / dusk / low sun / daytime, in rain / wet / cloudy / clear /
  fog, at a junction, at a speed breaker, in a log / simulation;
- **numbers**: "TTC under 1.5 s", "gap below 10 m", "above 40 km/h", "faster than 10 m/s", "deceleration at least
  4 m/s2" — a number with a unit and no name is taken by its unit (km/h: the ego's speed; s: TTC; m: gap); a bound
  written after the number also reads ("TTC 1.2 s or less", "gap 5 m or more");
- **time to collision** (or TTC) is always a TTC condition, never the collision type (fixed in 4.2: "time to
  collision under 1.5 s" used to add *type collision*): with a bound it is that bound, with a bare number after it
  ("time to collision 1.5 s") at most that number, and with no number at all *min TTC < 1.5 s* (the low-TTC
  threshold) — the understood line says which;
- **fidelity** (4.2): "in CARLA", "on the fake simulator", "UE5", "neural render", "kinematic screen" require runs of
  that level (`core.fidelity`); `--fidelity LEVEL` and the section's *Fidelity* box do the same.

The filter is always shown back (`understood as: …`) and every word outside the vocabulary is listed
(`not understood: …`) — never dropped silently. Moment types and numeric constraints are **required** of a hit; the
other terms are **scored**: hits matching all of them come first, then partial hits marked with what they miss
(`--strict` drops those), and within a level the more severe moment first.

Structured flags do the same without words and add to them:

```
falcon moments search my.falcon.json --type cut_in --actor two_wheeler --night --where "ttc<1.5"
```

## From a hit

- `falcon moments show WS ID` prints one moment in full.
- `falcon moments show WS ID --scenario NAME` writes it as a scenario through the incident path
  ([dashcam](dashcam.md)): the moment's window of recorded tracks, the parametric fit, `scenario.json` (a replay),
  `variations.json` (ranges around the extracted values, for `falcon search`) and `moment.xosc` (OpenSCENARIO 1.2) in
  `<workspace>/moments/scenarios/NAME/`.
- In the window, **Replay ▸ Moments**: the query box, the *Fidelity* box, the understood line, the hits with a
  severity meter and their run's fidelity; double-click (or *Open*) opens the run on the Data page at that time;
  *Open in Replay* shows it on the Replay page when that page can load a run in this build (an imported log loads
  there; otherwise the note says so); *Make a scenario…* and *Drive set…* (the runs of the full matches as one
  undoable drive set). Selecting a hit makes its run the window's active run; its context menu has *Send to ▸*.
- **Send to ▸ Moments** from any run row (4.2): the Replay page opens, the run is indexed if it is not yet (in the
  background, the same code as `falcon moments index --run`), and its moments are listed in time order;
  *Show all* goes back to the search.

Every indexed source records its **fidelity** (`runs.fidelity` in the store: a run folder's `run.json` level —
`unknown` before 4.2 —, `recorded` for an imported log or a near-miss ranking); every hit carries it. A store
written before 4.2 gains the column when it is opened, filled from the runs' `run.json` (no run is read again).

- In the **command palette** (Ctrl+K), a query the index understands (`cut-in ttc<1.5`, `pedestrian crossing`) lists
  its five best full matches under *Moments*; Enter opens the run on Data at the moment's time (a log's moment: the
  log on Replay). A query that means every moment (no recognised word) adds none; near-miss clips are not listed.

## Exposure

`falcon moments stats WS [--by tod|weather|world|source]` prints how often each type occurs per hour of indexed
driving and per 100 km, with an approximate 95 % Poisson interval (Byar's approximation; recalled, not checked
against the document; the tests compare it with the exact interval for a count of 10). The driving time is the
recorded time of the runs and logs that have tracks; near-miss rankings add events but no time and are listed apart.

These rates describe **this workspace** — mostly simulated scenarios chosen to be eventful — not real traffic. They
are not linked into the failure-rate or ODD-coverage estimates: an exposure there must come from real driving.

## Privacy

A run derived from dashcam footage (a route replay, a reconstruction, a near-miss ranking) is indexed locally like any
other and flagged private. The command line and the window show it to you, marked. The assistant's tools
(`runs_moments_search`, `runs_moments_stats`) read with private sources left out and report only how many were. A
scenario made from a private moment keeps the incident path's dashcam marker, so it stays private. Of a near-miss
ranking only the class, the window within the clip and the measures are taken — no clip name, date or position.

## Limits

- Query understanding is rule-based: a phrasing outside the vocabulary is not understood (and listed). There is no
  semantic matching.
- A straight corridor mislabels far actors on a curve; ranges are short for that reason, but a cut-in on a tight bend
  can be missed or a neighbour-lane vehicle taken for a lead.
- A run without `actors.jsonl` has only its scenario's own actors (no background traffic); a run without lanes has no
  lane departures, and jaywalking falls back to the junction heuristic or is not claimed at all.
- Red-light violations need a recorded light state; drives do not record one. Stop-sign violations are not detected.
- Actor classes are what the run recorded: a CARLA motorcycle and a scooter stand-in are both `motorcycle`.
- TTC is at constant velocity; a braking lead's TTC is therefore optimistic until it has slowed.
