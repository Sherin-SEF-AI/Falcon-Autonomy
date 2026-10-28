# Runtime ODD monitoring

A safety case covers an operational design domain (ODD); evidence gathered outside it is not evidence about it. The
runtime ODD monitor flags, while a run runs and afterwards over recorded runs, when its **conditions** (weather, light,
speed, road, place) or its **sensor data** leave the ODD (`falcon_autonomy/core/oddmonitor.py`, the workers'
`workers/oddwatch.py`).

## The ODD

The ODD monitored is the safety case's (`falcon.odd/1` with `dimensions`: [stack testing](stack-testing.md);
*Stack testing ▸ New from the example* writes one to the workspace as `odd.json`). The Loop's ODD has the same schema
string but another shape — worlds × weather entries ([loop](loop.md)) — and is **mapped**: place = its worlds and
their CARLA maps; each weather field = the hull, over its weather entries, of the entry's value (its preset's, or
fixed) and its range. The hull over-approximates: a combination of fields no single entry allows (one entry's rain
with another's fog) is still called inside; the mapped ODD says so in its `about`.

## Condition monitors

Exact, from the simulator's state at every step:

| Dimension | Runtime (workers) | Offline (recorded runs) |
|---|---|---|
| weather and light: any range named like a CARLA weather field (`precipitation`, `fog_density`, `wetness`, `cloudiness`, …) and `sun_altitude_deg` | `World.get_weather()` each step (a live weather event is seen) | the run's config; a frame's `odd.weather` when the worker recorded a change |
| `ego_speed_kmh` | the ego's **actual** speed (the director's in kinematic motion) | `speed_mps` of each frame |
| `road_type` | the map at the ego: `junction` (`is_junction`), `multi_lane` (a driving lane beside it in the same direction), `two_way` (one in the other direction), `straight` or `curve` (the lane's heading changes by less, or more, than 10° over the next 30 m) | the frames' `odd.road` when the worker monitored; else the safety case's heuristic from the scenario's requirements, and the result says so |
| `place`, `handedness`, `scenario`, `actors`, `traffic_vehicles` | the run's config | the run's config |

A range is inside when lo ≤ value ≤ hi; a categorical dimension when one of the run's values is listed; a set
dimension when all of them are. A dimension the state cannot give is reported as **not monitored**, never assumed
inside. Note that the safety case's *coverage* reads `ego_speed_kmh` as the scenario's set speed; the monitor reads the
speed actually driven, and `straight` and `curve` as observed — an ODD listing only `straight` flags every curve.

## Data monitors (out of distribution)

Per camera frame (every `odd_every`-th at runtime; every `--every`-th offline):

- **Appearance**: the 11 image statistics of [sim-to-real](sim-to-real.md) (CIELAB mean and spread, saturation,
  clipped shares, noise, sharpness, spectrum slope), standardised by the reference; the Mahalanobis distance under a
  Ledoit–Wolf shrunk covariance (O. Ledoit, M. Wolf, *A well-conditioned estimator for large-dimensional covariance
  matrices*, J. Multivariate Anal. 88, 2004). About 24 ms per 1280 × 720 frame.
- **Features** (optional, only when the pinned ResNet-50 extractor of the sim-to-real tools is downloaded): the mean
  cosine distance to the 5 nearest reference frames in its pooled feature space.

A frame is out of distribution when any score exceeds its threshold; between scored frames the last frame's flag
holds (the time outside counts every step it holds). A reference that has a feature monitor used where
the weights are not downloaded falls back to appearance alone, and says so.

### Calibration

`falcon odd calibrate --in RUN... [--out-of NAME=RUN[,RUN]...] --far A --out reference.json` builds the reference from
in-ODD drives. Frames of one drive are not independent, so the unit is the **drive**: leave-one-drive-out
cross-fitting. Each in-ODD drive is scored by a reference fitted on the other drives; the threshold is the conformal
quantile of these out-of-fold scores for the target false-alarm rate A (the ⌈(n + 1)(1 − A)⌉-th smallest of n), so the
drive-to-drive variation is part of it; the final reference is fitted on all drives. The false-alarm rate is then
**measured**, nested: the whole procedure is repeated without each drive in turn, and that drive's frames above the
threshold so obtained are counted — a drive never takes part in its own threshold or reference. The detection rate is
measured on each `--out-of` set. Both carry Wilson 95 % intervals **over frames**, which are optimistic (frames of a
drive are correlated); the per-drive counts are reported beside them. At least three in-ODD drives are needed; with
fewer, each drive is cut into three contiguous blocks used as the units, and the report says the rate is then
optimistic.

### Measured

**Real frames.** In-ODD: three real CARLA 0.9.16 drives of Town10HD at clear noon, 1280 × 720 (two of the
milestone-5 drives and the ground-truth check's Town10 drive; every 2nd frame: 80, 120 and 60 frames). Target
false-alarm rate 5 %.

| Out-of-ODD set (80 or 60 frames) | Appearance only: detected | With ResNet-50 features |
|---|---|---|
| fog, visibility 350 m (fog_density 20) — **synthesised**, see below | 62 of 80 = 0.78 (95 % 0.67–0.85) | 62 of 80 |
| fog, visibility 140 m (fog_density 50) — **synthesised** | 80 of 80 = 1.00 (0.95–1.00) | 80 of 80 |
| rain on the lens, 25 mm/h — Falcon's camera physics on the same frames | 22 of 80 = 0.28 (0.19–0.38) | 22 of 80 |
| another place: the imported central-Bengaluru world, clear noon (60 frames) | 60 of 60 = 1.00 (0.94–1.00) | 60 of 60 |
| **false alarms**, in-ODD drives, nested | **0 of 260 = 0.000 (0.000–0.015)** | 33 of 260 = 0.127 (0.092–0.173), all on one drive |

The fog was synthesised on one in-ODD drive's frames by Koschmieder's law from each pixel's range label
(transmission e^{−3.0 r / V} towards a grey airlight) — not CARLA's fog renderer, which was not run for this (the CARLA
server was in use for other work). The appearance monitor's threshold came out conservative (no false alarm in 260
frames against a 5 % target): with three drives the out-of-fold scores are large, because the drives differ more from
each other than frames within a drive do (for example the mean saturation was 0.135, 0.208 and 0.212 by drive, with
spreads of 0.01–0.02 within each). The price is sensitivity: light rain on the lens is mostly not detected. The
feature monitor added nothing here and raised the false alarms; it is not on by default. An earlier single-split
calibration (one drive to fit, one to set the threshold, one to test) measured 58 false alarms in 80 frames: with few
drives a reference fitted on one drive calls every other drive out of distribution.

**Fake carla** (the tests): four clear drives against a fog drive (fog_density 50, rendered by the fake with CARLA's
visibility relation): at least 90 % of the fog frames detected; the measured false-alarm rate is reported, not
asserted, because four 20-frame drives cannot bound it tightly.

## Where it runs

A run's config may carry `odd` (either format), `odd_reference` (a file from `falcon odd calibrate`), `odd_every`
(default 10) and `odd_workspace` (for a feature monitor). Without `odd` nothing is monitored or recorded.

- **Drives** (`falcon drive`, data engine, scenario runs): each frame's `frames.jsonl` record gets
  `"odd": {"in", "why", "road", "weather" (when it differs from the config's), "score" (a scored frame)}`; `run.json`
  → `odd_monitor` (the timeline, the time outside per reason); `odd-monitor.json` beside the run.
- **Live**: the same per step; each frame sent to the page carries `odd`, and the view shows a small amber line *▲
  outside the ODD: …* only while outside; the page's measurements list *ODD*.
- **Gym episodes**: each step's `info["odd"]`; each episode's summary in `episodes.jsonl` gets `odd_monitor`.
- **Stack runs**: `frames.jsonl`, `run.json`, `odd-monitor.json` (the data monitor scores the first bridged camera).
- **HIL sessions**: the condition monitors per step in the session's records and `run.json` (one simulator query per
  step, the weather — it shows in the latency measured).
- **Offline**: `falcon odd monitor RUN... --odd FILE [--reference REF] [--every N]` writes `odd-monitor.json` into each
  run and prints the time outside and why.
- **Safety-case report**: a section *Time outside the ODD* — per scenario of the suite, the runs outside, the time and
  share outside, and why — from the condition monitors over each run's frames; any time outside is also a residual risk
  ("that evidence is not about the ODD"). `safety-case.json` → `odd_monitor`.
- **Stack testing page**: *Outside the ODD* in the suite's summary (cached per run), for the ODD chosen there.
- **Loop**: the evaluation cells (world × weather entry) checked against the workspace's `odd.json`: *Safety-case ODD*
  says which cells are outside or partly outside (a ranged field with one end outside), and the weak-cell table marks
  them *⚠ outside the ODD*.

`odd-monitor.json` (`falcon.odd_monitor/1`): `steps`, `dt`, `time_s`, `time_outside_s`, `share_outside`,
`outside_by_dimension_s`, `data` (frames scored, flagged), `segments` (t0, t1, inside, the dimensions outside, the
reasons, whether the data monitor flagged it), `monitored`, `not_monitored`, `notes`.

## Limits

- The data monitor is calibrated on what it was given: three drives of one town at one weather are a narrow
  reference, and its false-alarm rate is measured over correlated frames. A reference must come from the ODD's own
  training or evaluation drives, with the same rig (resolution, compression and lens change the statistics).
- An appearance monitor detects shifts of the image's statistics, not hazards: dense fog and a different city were
  detected, light rain on the lens mostly not; a scene that looks ordinary but is outside the ODD in a way the
  conditions do not name will not be flagged.
- The fog of the measurement was synthesised, not CARLA's; detection of CARLA's own fog on real drives was not measured.
- Road features come from CARLA's map at the ego (one waypoint, 30 m ahead): lane counts beside the ego only, no
  speed limits, no surface type. Offline runs recorded without the monitor get the scenario heuristic for road type.
- The Loop mapping over-approximates the Loop's ODD (the hull of its weather entries).
