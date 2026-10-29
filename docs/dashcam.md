# Your own dashcam

`falcon dashcam` (and Training ▸ *Sim-to-real translation* ▸ *Or your own dashcam*) reads a consumer dashcam's MP4
clips: the GPS track the camera embeds, frames stamped with the position they were taken at, which clips drive inside
an imported world, and an image set that Falcon's [gap](sim-to-real.md) and [translator](translation.md) read as a real
set (`dashcam:FOLDER`). Code: `falcon_autonomy/core/dashcam.py` (its docstring is the reference; its public functions
are a stable API that the reconstruction's dashcam odometry imports).

```sh
falcon dashcam list [--root ~/Videos/dashcam-videos]          # clips: fixes, distance, speed, sun altitude
falcon dashcam gps CLIP [--json | --raw]                        # one clip's decoded track and its plausibility check
falcon dashcam extract --name NAME [--day] [--max-frames N] [--rate 2] [--size 640x360] [--crop-bottom 0.06] \
    [--min-speed 2] [--require-gps] [--holdout N --split train|holdout] [--include GLOB]
falcon dashcam match --project P.falcon.json --world W [--out DIR]
```

## Privacy

Dashcam footage shows where and when its owner drove — the track, the timestamps, often the home and work places, and
other people's faces and number plates. Falcon reads clips in place and writes frames only where you say
(`~/FalconWorkspace/realsets/<name>` by default — not `/tmp`, which is a small RAM-backed folder); nothing is uploaded,
and nothing of the footage enters the project or its repository (the tests make their own synthetic clips). The camera
verified here also **burns its date, speed and position into every frame** (a text line at the bottom); `--crop-bottom
0.06` cuts it off. A translator trained on your frames, and every image it makes, carry them: `translator.json` says so
(terms: "your own dashcam footage"); keep both private unless you mean to share that.

## The clips

Files named `YYYYMMDDHHMMSS_NNNNNN<F|R>.MP4`: the start in **local time** (`TZ_OFFSET_H`, 5.5 h by default), a running
number, the front or rear camera; folders `Movie_F` (driving), `Park_F` (event/parking-mode clips — on this camera they
hold driving too) and `Photo_R`. The camera verified: HEVC (H.265) 3840 × 2160 at 30 frames/s, 60-second clips of
1800 frames, constant frame rate, a key frame every 0.5 s; AAC audio. The MP4's `creation_time` is the clip's *end* in
local time mislabelled as UTC — not used.

## The GPS format, verified

The track is in `freeGPS` blocks (the Novatek family; this variant signs them `YOUQINGGPS`). Nothing was guessed: the
layout below was read from the files and each field checked against independent evidence.

**Where.** The `moov` box holds a box `gps ` whose body is u32 version/flags (0x00000101), u32 count, then count
(u32 offset, u32 size) pairs, big-endian — **one per second of video** (60 per clip). Each points to a 16 384-byte block
inside `mdat`: u32 size, `freeGPS ` (so the block parses as an MP4 `free` box), u32 little-endian payload length
(0x90 = 144), the payload. A second without a fix is a block of zeros after the header.

**The payload** (little-endian; offsets from the signature):

| offset | type | field | evidence |
|---|---|---|---|
| 0 | 12 bytes | `YOUQINGGPS\0\0` | constant in every block with a fix |
| 24, 28 | f32, f32 | latitude, longitude as ddmm.mmmm (degrees · 100 + minutes) | decode to plausible positions along the roads driven; consecutive fixes move continuously (no implied speed above 160 km/h in 6 271 pairs) |
| 32 | 3 × i32 | hour, minute, second, **UTC** | + 5 h 30 min = the camera's own clock (offset 120) within ±1 s, and = the file name's time + the block's index (+0 or +1 s) |
| 44 | 3 × i32 | year − 2000, month, day (UTC) | the file names' dates |
| 56 | 3 chars | status `A`/`V`, `N`/`S`, `E`/`W` | always `A N E` here (a missing fix is an empty block, never `V`) |
| 64 | 2 × f64 | **not identified** | the first varies at random from second to second, the second grows slowly while moving; not used |
| 80 | f32 | **speed over ground, knots** | the distance between consecutive one-second fixes ÷ (mean reported speed in knots → m/s): median **1.002** (10th–90th percentile 0.969–1.041, 6 271 pairs above 5 m/s); read as km/h the ratio would be 1.86. The camera's burned-in line shows the same speed in km/h (72 km/h for a fix of 38.9 kn) |
| 84 | f32 | course over ground, degrees from north | the heading between consecutive fixes minus it: median 0.00° (10th–90th percentile −1.5° … +1.5°) |
| 88 | 16 chars | `1234567890123456` | constant (a placeholder serial) |
| 104 | 4 chars | `YSKJ` | constant |
| 108 | 3 × i32 | small integers (−6 … 6) changing every second | probably the G-sensor; **not identified**, not used |
| 120 | 6 × i32 | hour, minute, second, year, month, day of the camera's clock (local time) | equals the file name's time + the block index (+0/+1 s) |

On the user's 262 clips: 15 539 blocks, 15 033 fixes after the time base (414 repeated UTC seconds dropped), 91
empty blocks, two clips with no fix at all, and **one index entry that points into video data** (one clip's block 6)
— rejected by the checks
(signature, ranges, minutes < 60), counted as unrecognised, never decoded as a position. Float32 coordinates carry
≈ 0.2 m (latitude) and ≈ 0.9 m (longitude) of quantisation, which is why a single pair's distance/speed ratio scatters
by a few per cent.

**Uncertain.** The two f64 and the three small integers are not identified. Whether other firmware of the same family
uses the same offsets is not known: a block that fails the checks is skipped and counted, so a different layout shows
up as "unrecognised", not as wrong positions.

## The time base

Block *k* is written into `mdat` right after the video packets of 0 … *k* + 0.93 s (measured from the packets'
positions on 31 clips), so block *k* describes video second *k*. Falcon puts a fix at video time **t = UTC − t0**, with
t0 the clip's median of (UTC*k* − *k*): the GPS's own time, so a repeated or skipped second (≈ 3.7 % of blocks, against
the block index) lands where it belongs; a repeated UTC second keeps its first fix. A frame's position is interpolated
linearly between the fixes around its time (the course on the circle); a frame more than 2.5 s from any fix gets none.

Checked against the video itself: on 20 stop-and-go clips the GPS speed was cross-correlated with the image motion
(median optical-flow magnitude at the image sides, 5 frames/s, lags −8 … +8 s). The 10 clips with a clear peak
(correlation ≥ 0.5, peak within ±5 s) put it at −1.4 … +2.4 s, median 0.0 s; four more with correlation ≥ 0.5 peaked
at +6.6 … +8.0 s, at the edge of the search, on flat curves (their correlation at lag 0 is within 0.06–0.18 of the
peak); six had no usable peak. Traffic ahead stops and starts with the car, so the curves are broad: this supports the
time base to about ±2 s on the clear clips and does not rule out a larger offset on some. **The sub-second offset between a fix
and a frame is not known from the file** (GPS latency is not recorded): about ±0.5 s, i.e. ±7 m along the track at
50 km/h. One caution from the burned-in line: on the frame checked by eye (one clip, frame 600) its clock
agrees with the time base within 1 s, but the position it prints is the fix of 5–6 s earlier — the camera's overlay
lags; the embedded fixes do not (their motion agrees with the video, above).

## Frames and image sets

ffmpeg decodes (`-hwaccel cuda`, NVDEC, when it works; checked once per codec), scales with area averaging and
returns RGB; a frame's index *n* gives its time *n* / fps (constant frame rate, checked from the header; ffmpeg's own
timestamps confirm it). At a rate whose step falls on the key frames (2 frames/s on this camera) only key frames are
decoded. Measured on 4K HEVC clips (RTX 5080 NVDEC; 24 CPU threads):

| 2 frames/s at 640 × 360 out of 4K HEVC | video | wall time |
|---|---|---|
| NVDEC, key frames only | 60 s | 1.0 s |
| NVDEC, every frame decoded | 60 s | 3.9 s |
| CPU, key frames only | 60 s | 2.4 s (6.5 CPU-seconds) |
| NVDEC, every frame decoded | 20 s | 1.9 s (2.4 CPU-seconds) |
| CPU, every frame decoded | 20 s | 3.3 s (21 CPU-seconds) |

Whole extractions (the training set below, key frames on NVDEC, JPEG writing included): 4000 frames out of 209
clips (5140 s of video between each clip's first and last kept frame) in 123 s — 32.8 frames/s, 42 s of video per
second; the held-out set, 600 frames out of 53 clips in 35 s.

An image set:

```
<name>/
  frames/<clip>_<n>.jpg     640 × 360 (or as asked), JPEG quality 95 (--png for lossless)
  frames.jsonl              per frame: image, clip, frame, t (video s), utc, local, lat, lon, speed_kmh, course_deg,
                            gps ("interpolated" | "nearest" | null), sun_deg
  run.json                  provenance: arguments, settings, seed, counts (candidates, static, dark, without a fix,
                            written), timings, the decoder used, and each clip's fingerprint
```

Filters, in order: frames on the rate's cadence; **static** — slower than `--min-speed` (2 km/h) at the frame's
interpolated speed; **dark** — with `--day`, the sun lower than `--min-sun` degrees at the frame's position and time
(NOAA's formulae, ≈ 0.5°); `--require-gps` drops frames without a position (otherwise they are kept, without one);
`--max-frames` then takes that many evenly across everything left (one per equal stretch, at a seeded place in it).
`--holdout N --split holdout|train` splits by clip (every N-th clip held out), so a test set shares no clip with a
training set. **Fingerprints**: hashing 59 GB on every extraction would take minutes, so each clip is identified by
its size, modification time and the SHA-256 of its first MiB (which holds the `moov` box with the GPS index); an
edited clip almost surely changes one of them, but this is not a full-content hash.

`dashcam:FOLDER` is a [gap](sim-to-real.md) and [translation](translation.md) real set: no labels; each frame's GPS,
speed, clip, UTC and sun altitude (night = below 0°) as its conditions.

## The footage on Falcon's map

`falcon dashcam match --project P --world W` projects every fix into the imported world's frame — its transverse
Mercator about the import centre (`core.osm.local_projection`, the `+proj=tmerc` string in the OpenDRIVE header):
x east, y south (CARLA's), yaw = course − 90° — and reports per clip the fixes inside the import box, the distance to
it, and, for every stretch within 50 m of an imported driving lane, the lateral distance to the nearest lane centre
and [log replay](log-replay.md)'s map-match verdict on the track interpolated every 0.2 s. It also proposes the 1-km
square (sliding on a 250 m grid) that holds the most moving footage — where an import would match the most of it.

**The user's footage against `w-blr`** (Central Bengaluru, 1 km about 12.9716° N, 77.5946° E — MG Road / Cubbon Park):
**no clip drives inside it, and none comes within 50 m of its lanes; every fix is several kilometres outside the
box.** The daylight footage is on highways well outside the imported area; the clips in the city were recorded
**after sunset**. So the real footage does **not** show the simulated streets: to compare a clip with the same street
simulated, import a world there (Worlds ▸ Import, centre and size from `match`, which proposes the densest square of
moving footage).

## Translation to your own roads, measured

Falcon's [translator](translation.md) (CUT, the frame's label as an input, default settings: λ_NCE 10, 12 000
iterations of batch 4, 192-px crops, seed 0) trained towards the dashcam, on the reference machine (RTX 5080),
2026-10-29.

**Data.**
- *Real, training*: 4000 frames, 640 × 338 (the bottom 6 % — the burned-in text line — cut), daytime (sun ≥ 5°) and
  moving (≥ 2 km/h), spread evenly over 209 clips (`--holdout 5 --split train`).
- *Real, held out*: 600 frames of the same kind from the other 53 clips (every fifth clip) — no clip shared.
- *Simulated*: two drives recorded for this on Falcon's own CARLA 0.9.16 server (8 GB cap, a 640 × 360 front camera
  with its semantic label, 60 s at 10 frames/s, ClearNoon): Town10HD (seed 11; 30 cars; the car waits at a red light
  for about 25 s of it) and the imported Central Bengaluru `w-blr` (seed 7; the Indian mixed traffic profile). Every
  second frame (300 + 300) trained the translator; **the gap and the drift below are measured on all 600 frames of the
  same two drives** — the simulated side is not held out (the real side is).

The probe drift (eight fixed simulated frames, every 1000 iterations) stayed at 0.00–0.05 for the first 6000
iterations (one 0.16) and then climbed — 0.34, 0.32, 0.42, 0.44, 0.50, **0.54** at 12 000. Because of that curve a
second translator was trained with **6000 iterations** (otherwise the same; its learning rate decays over its own
second half, so it is not the first one's iteration 6000): probe drift 0.05 0.01 0.16 0.02 0.01 **0.03**.

| | default (12 000 it.) | 6000 iterations |
|---|---|---|
| training time, images/s (median), peak GPU memory | 997 s, 48.5, 1927 MiB | 637 s (the GPU shared with a test run) |

**Drift and refusals** (threshold 0.35, the translated runs' `translation.jsonl`; medians):

| | structure | edge recall | edge precision | worst object box | drift | refused |
|---|---|---|---|---|---|---|
| Town10HD, default | 0.907 | 0.973 | 0.954 | 0.531 | 0.463 | **400 of 600** |
| Bengaluru, default | 0.944 | 0.808 | 0.332 | 0.621 (46 frames with objects) | 0.668 | **600 of 600** |
| Town10HD, 6000 it. | 0.988 | 0.998 | 0.992 | 0.987 | 0.013 | 0 of 600 |
| Bengaluru, 6000 it. | 0.996 | 0.990 | 0.844 | 0.985 | 0.156 | 0 of 600 |

The default translator repaints CARLA's vehicles (in every one of Town10's 400 refusals the worst object box is the
lowest of the four measures) and,
on the bare generated Bengaluru world, paints fields and roadside clutter onto the flat ground (edge precision 0.33:
"added structure") — every Bengaluru frame is refused and kept raw. The 6000-iteration translator changes tone and
texture and keeps the structure by the measures — but look at its Bengaluru frames: it lays soft cloud-like blotches
on the empty road surface that stay under the threshold (drift 0.16–0.27). A drift under the threshold is not proof
that a frame is right.

**The gap to the held-out dashcam frames** (`falcon translate check --features --images 300`: FD and KD on ResNet-50
features, 150 simulated frames against one random half of the 300 real ones, the floor between the two real halves;
seed 0; one training of each, so training noise is not measured):

| simulated frames | FD raw → translated (floor) | KD raw → translated (floor) |
|---|---|---|
| Town10HD, default, refused frames raw (the label-safe run: 400 of 600 raw) | 151.2 → 142.1 (31.2) | 0.142 → 0.122 (0.000) |
| Town10HD, default, every translation kept (`--on-drift flag`) | 151.2 → 153.8 (31.2) | 0.142 → 0.138 (0.000) |
| Town10HD, 6000 it. (none refused) | 151.2 → 148.0 (31.2) | 0.142 → 0.138 (0.000) |
| Bengaluru, default, refused frames raw (all 600 raw) | 218.9 → 218.9 (31.2) | 0.226 → 0.226 (0.000) |
| Bengaluru, default, every translation kept | 218.9 → 174.7 (31.2) | 0.226 → 0.171 (0.000) |
| Bengaluru, 6000 it. (none refused) | 218.9 → 190.6 (31.2) | 0.226 → 0.188 (0.000) |

Appearance (medians; real dashcam / raw → translated, shift = distance over the real set's IQR):

| | lightness L | contrast L_std | saturation | blue–yellow b | sharpness |
|---|---|---|---|---|---|
| dashcam (held out) | 49.8 | 17.6 | 0.124 | 0.30 | 2.81 |
| Town10HD raw | 59.7 | 18.0 | 0.144 | 4.93 | 2.74 |
| → default, all kept | 40.5 | 21.1 | 0.219 | 5.39 | 3.15 |
| → 6000 it. | 53.1 | 17.9 | 0.143 | 3.01 | 2.83 |
| Bengaluru raw | 81.2 | 8.0 | 0.063 | 1.49 | 1.24 |
| → default, all kept | 56.2 | 14.0 | 0.195 | −1.03 | 2.94 |
| → 6000 it. | 74.5 | 8.3 | 0.070 | −0.75 | 1.32 |

What this says:
- **The simulated worlds are far from the dashcam** — FD 151 (Town10HD) and 219 (Bengaluru) against a real-vs-real
  floor of 31 — and the generated Bengaluru world is further than Town10 although it is "Bengaluru": its untextured
  ground and empty roadsides (lightness 81, contrast 8, no texture) are what dominate, not the city.
- **Label-safe translation moves the gap little on Town10**: FD −6 % (default, refused frames raw) and −2 % (6000
  iterations, none refused). The 6000-iteration translator fixes most of Town10's colour cast (b 4.9 → 3.0, lightness
  59.7 → 53.1; real 0.3 and 49.8). On Bengaluru the default leaves nothing (every frame refused); the 6000-iteration
  translator lowers FD by 13 % with no refusal — partly through the blotches on the road described above.
- **The large drops are hallucination**: Bengaluru's FD falls 20 % only when the default translator's invented fields
  and roadside are kept — exactly the frames whose labels would be wrong. Town10 with every translation kept gets
  *worse* (FD 153.8): the default translator overshoots (lightness 40.5, saturation 0.219 against the real 49.8 and
  0.124).
- The rest of the gap is content: daytime Indian highways and towns under heavy cloud (trucks,
  two-wheelers, medians, open fields, a wide lens and the bonnet), against CARLA's American town and a bare imported
  street network. A label-preserving translation cannot add that; a better Bengaluru world (buildings, roadside
  assets, textures) and real footage of the same streets would.

**Examples** (raw | translated, the drift and whether it was kept), written beside the sets, not in the repository:
`~/FalconWorkspace/realsets/dashcam-translation/examples/` — `town10_*` and `blr_*` (default translator; refused frames
re-run through the network so its invention can be seen), `6k_*` (6000 iterations), and
`real_heldout_reference.jpg` (four held-out dashcam frames). One of the default translator's *kept* Town10 frames
(`town10_step0121_drift0.09_kept.jpg`) visibly smears the far street — another reminder that the drift measures are
heuristics.

Reproduce (paths as on the reference machine):

```sh
falcon dashcam extract --name dashcam-day-train   --holdout 5 --split train   --day --min-sun 5 --max-frames 4000 --crop-bottom 0.06
falcon dashcam extract --name dashcam-day-holdout --holdout 5 --split holdout --day --min-sun 5 --max-frames 600 --seed 1 --crop-bottom 0.06
W=~/FalconWorkspace/realsets/dashcam-translation
falcon translate train --out $W/translators/dashcam-cut --sim "$W/sim/town10?every=2" --sim "$W/sim/blr?every=2" \
    --real dashcam:~/FalconWorkspace/realsets/dashcam-day-train --max-images 4000 --seed 0      # [--iterations 6000]
falcon translate apply --model $W/translators/dashcam-cut --run $W/sim/town10 --out $W/translated/town10-tr
falcon translate check $W/translated/town10-tr --real dashcam:~/FalconWorkspace/realsets/dashcam-day-holdout \
    --features --workspace $W --images 300 --out $W/checks/town10
```

## Limits

- One camera family verified (YOUQING freeGPS, HEVC 4K). Another layout is skipped and counted, not decoded.
- The GPS-to-frame offset is ±0.5 s by the file's layout and supported to about ±2 s by the video's motion on clear clips;
  positions are a consumer GPS's (no accuracy is recorded), interpolated linearly at 1 Hz.
- The burned-in text and the bonnet are part of every frame unless cut (`--crop-bottom`); the camera's wide lens
  differs from the simulated camera's 90° (not corrected).
- Clips without a parsable name have no start time; a clip whose `moov` box is missing (still being written) is
  skipped.
- Fingerprints are partial hashes (above).
- The translation figures are single trainings (seed 0) measured on the same simulated drives that trained them, with
  FD and KD on 150 simulated against 150 real frames: their noise is not measured.
