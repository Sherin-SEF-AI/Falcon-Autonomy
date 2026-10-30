# Synthetic audio: a directional microphone array

CARLA 0.9.16 has no audio. On Indian roads much of the negotiation is heard before it is seen — horns asking the way or
announcing an overtake, a siren behind the traffic, a two-wheeler's engine closing from the side — so Falcon derives a
**synthetic** multi-channel recording of a microphone array on the ego from what a drive recorded
(`falcon_autonomy/core/audio.py`, numpy only). Every sound is **procedural**: emission models with cited sources, and
heuristics labelled as such. No sample, no recording, no measured vehicle sound is used. It is a test signal for
direction-of-arrival, detection and fusion pipelines with exact ground truth — not a substitute for real recordings.

## Making it

- **After a drive** (the way to do it): `falcon derive audio RUN [--out DIR] [--option KEY=VALUE]...` — `RUN/audio` by
  default, which also records the parameters in the run's `run.json` (`derived.audio`). Data ▸ Browse ▸ **Derive
  audio** (or the run list's right-click menu) runs the same command as a background job for every selected run.
- **With the drive**: `falcon drive PROJECT … --audio` derives it once the drive has finished.
- **The array** is the rig's `audio_array` sensor (Rigs ▸ Add ▸ *Microphone array (synthetic)*): its pose on the
  vehicle, and either `mics` microphones on a circle of `radius` m in the array's x–y plane (mic 0 ahead, then towards
  +y, the right) or explicit `mic_positions` ([[x, y, z], …] m in the array's frame). Defaults: 8 microphones, radius
  0.2 m; a rig without one gets the default array 1.55 m above the vehicle's origin (on an MKZ's roof). Nothing is
  spawned in CARLA for it. Option `array` overrides the rig's (`--option 'array={"mics": 4, "radius": 0.3}'`).
- **Options** (all recorded in `audio.json`): `temperature_c` 20, `humidity_pct` 70, `pressure_kpa` 101.325,
  `air_absorption` true, `ground` true, `ground_reflection` 0.95, `occlusion` true, `max_range_m` 200 (sources farther
  from the ego are not rendered; they fade over 0.3 s at the boundary), `audible_db` 30 (the ground truth's threshold),
  `mic_self_noise_db` 29, `horns` auto (`indian` on a left-hand or Indian profile, else `sparse`; or `none`),
  `horn_events` (explicit `[{"actor": ID, "start_s": S, "dur_s": D}]`), `sirens` on (or `off`, or `moving`: only while
  the vehicle moves), `include_ego` false, `seed` (the drive's), `format` float32 (or `pcm24` with `full_scale_pa`
  200), `sample_rate` 48000, `duration_s`, `doa` true (run the SRP-PHAT check). An unknown option is refused.
- **Output**: [dataset format](dataset-format.md#synthetic-audio-derived) — `audio.wav` (pascals), `ground_truth.jsonl`
  (per step: every audible source's id, class, kind, azimuth and elevation in the array frame, distance, received
  level, Doppler factor), `events.jsonl` (horns and sirens), `audio.json`, `doa.json`.

## The model

### Sources
Each source is a point; "at 1 m" means the free-field pressure 1 m away.

| source | model | numbers and where they come from |
|---|---|---|
| rolling + propulsion noise of every motor vehicle | **CNOSSOS-EU** road-traffic emission: per octave band (63 Hz–8 kHz) L_WR = A_R + B_R·lg(v / 70 km/h), L_WP = A_P + B_P·(v − 70)/70, summed energetically; Lp(1 m) = L_W − 10·lg 4π | coefficients of Directive (EU) 2015/996 Appendix F as amended by Commission Delegated Directive (EU) 2021/1226, transcribed from NoiseModelling's `RoadCnossos_2020.json` (Université Gustave Eiffel); categories 1 (cars, vans), 2 (two-axle buses, the ambulance), 3 (lorries, the fire engine), 4b (two-wheelers > 50 cc: propulsion only). Below 20 km/h propulsion is held at its 20 km/h value (the model's range is 20–130 km/h; Falcon's choice) |
| acceleration | IMAGINE's ΔL_WP = C_P·a, a ≥ −1 m/s² (C_P·(−1) below), a ≤ 2 (cat. 1), 1 (cat. 2, 3), 4 (cat. 4) m/s² | Peeters & van Blokland (2007), *The noise emission model for European road traffic*, IMAGINE Deliverable D11, §4.3.1. D11 gives C_P per band only as a graph without an absolute scale; Falcon uses **3.2 dB per m/s² in every band**, the slope of D11's figure 18 (category 1, microphone 1) — **heuristic** in its spectral shape. a is the speed's derivative over 0.5 s |
| spectrum inside a band | the eight band densities interpolated in lg f, then each band scaled to its exact power; steep (f⁴) roll-off outside 44.7 Hz–11.3 kHz | Falcon's |
| engine orders | half of the 63–1000 Hz propulsion power moved into the first 24 harmonics of the firing frequency (rpm/60 × cylinders/2) up to 2 kHz, amplitudes ∝ 1/k, random start phases | **heuristic**; rpm from speed through a gear model per class (petrol 4-cylinder cars, diesel 4 vans, diesel 6 buses and lorries, single-cylinder two-wheelers 2500–6500 rpm, a Bajaj RE-like single for the autorickshaw 2000–4500 rpm) |
| autorickshaw (stand-in) | no CNOSSOS class: category 4b propulsion (a single-cylinder engine) + category 1 rolling × 3/4 (three tyres) | **heuristic** |
| source height | 0.05 m; two-wheelers and the autorickshaw 0.30 m | CNOSSOS-EU's single source height; IMAGINE D11 §2.1.3 (two-wheelers: one source at 0.30 m) |
| sirens | CARLA's `vehicle.ford.ambulance` (wail 8 s / yelp 4 s alternating), `vehicle.carlamotors.firetruck` (wail), `vehicle.dodge.charger_police*` (yelp): a band-limited square-ish wave (odd harmonics, 1/k, below 12 kHz) sweeping 650–1550 Hz, wail 12 cycles/min, yelp 200 cycles/min; 123 dB(A) at 3 m; on the box's top centre | KKK-A-1822 (US federal ambulance specification): fundamental 500–2000 Hz, wail 10–18 and yelp 150–250 cycles/min, at least 123 dB(A) at 3 m on axis. The sweep's end points are Falcon's choice inside that range; Indian siren specifications were not found and are not claimed |
| horns | a pulse-like tone (harmonics 1/k to 6 kHz): cars and vans 420 + 510 Hz at 98 dB(A) 7 m ahead, buses and lorries (air horn) 300 + 370 Hz at 105 dB(A), two-wheelers 400 Hz and autorickshaws 380 Hz at 88 dB(A); at the box's front, 0.6 m up (two-wheelers 0.8 m) | AIS-014 §6.1.2 (India: measured 7 m in front, 83–112 dB(A) for two- and three-wheelers ≤ 7 kW, 93–112 dB(A) for the rest); the notes and the levels inside those ranges are **Falcon's** |
| silent | bicycles (and the cycle-rickshaw stand-in), walkers, props | — |

**When horns sound** — CARLA has no horn, so a seeded rule set decides (**heuristic**, labelled in every event): at
each simulation step, a vehicle (not the ego) honks with probability 1 − exp(−λ·dt), λ the sum of the rates of the
rules that hold, times its class's factor and the profile's.

| rule | condition | rate (honks/s) |
|---|---|---|
| `slow_leader` | moving > 3 m/s, a vehicle 3–25 m ahead within 1.8 m of its line, closing by > 2 m/s (asking the way, an overtake) | 0.35 |
| `pedestrian_ahead` | moving > 2 m/s, a walker 2–20 m ahead within 3 m | 0.5 |
| `crossing_traffic` | moving > 1 m/s, a vehicle 3–20 m ahead within 6 m heading 45–135° across, moving > 2 m/s | 0.3 |
| `queue` | stopped ≥ 3 s with a vehicle 0–8 m ahead in line | 0.06 |

Class factors: two-wheeler and autorickshaw 1.0, bus and lorry 0.8, car and van 0.5. Profiles: `indian` 1.0, `sparse`
0.15, `none` 0. After a honk the vehicle waits 2.5 s; a honk lasts 0.12–0.45 s, and 35 % are a double toot. Every draw
comes from the vehicle's own generator (seed, actor id), so adding a vehicle never changes another's horns. Scenario
actions have no horn action yet; explicit events go in `horn_events`.

### Propagation, from each source to each microphone
- **Retarded time.** The microphone at time t hears what the source emitted at τ, with t − τ = |m(t) − p(τ)| / c. τ is
  solved by Newton's method on a 1 ms grid for every microphone (and for its ground image), and interpolated linearly
  to every sample; the emitted signal is read at τ with 4-point cubic interpolation. Source and ego paths are C1 cubics
  through actors.jsonl's and frames.jsonl's poses (every simulation step), extended backwards at constant velocity for
  the sources present at the first step (so the start is not silent while the first sound travels). **The Doppler
  shift is not applied anywhere**: it is what reading the emission at τ(t) does; its instantaneous factor dτ/dt =
  (1 − n·v_mic/c)/(1 − n·v_src/c) is written to the ground truth, not used. c = 343.2·√(T / 293.15 K) m/s.
- **Spreading** 1/r (spherical). The convective amplification of a moving source is left out (≤ 1 dB at 20 m/s).
- **Air absorption**: ISO 9613-1:1993's pure-tone coefficient α(f, T, RH, p), applied per frequency bin on the emitted
  signal's 21 ms frames at the frame's source–array distance (the microphones' separation changes it by < 0.1 dB at
  8 kHz).
- **Ground**: one specular reflection — an image source mirrored in the flat plane through the ego's ground — with a
  real reflection factor (0.95: hard, asphalt-like; **heuristic**: no ground impedance, no spherical-wave ground wave),
  its own retarded time and spreading, so its comb filter and Doppler emerge too.
- **Occlusion**: when the line from a source to the array passes through another vehicle's box (actors.jsonl), the
  diffraction over that box's top edge attenuates it by ISO 9613-2 eq. 14, D_z = 10·lg(3 + 20·z/λ) with z the path
  difference over the edge, capped at 20 dB (single diffraction, the most obstructing box). A slab test per frame per
  vehicle — cheap. **Buildings are not modelled**: the run folder does not hold their geometry.
- **Not modelled**: reverberation and reflections from buildings or vehicles, the ego's own body (no shadow or
  reflection at the roof), wind and turbulence, refraction, source directivity (sirens and horns radiate equally all
  round: an overestimate beside and behind), the ego's own noise (`include_ego` adds it as a source), wind noise on the
  microphones. The microphones' self-noise is added: uncorrelated white noise, 29 dB SPL per channel (a typical MEMS
  microphone's; Falcon's default).
- **Breaks**: a source's path is split where the traffic keeper moved it (frames.jsonl `traffic_respawns`) or where it
  moved farther in one step than its speed allows; each piece fades in and out over 20 ms. When the road-end guard
  moved the ego (`teleport`), nothing emitted before the jump is heard after it.

### Determinism
The same run, options and seed give the same samples bit for bit (measured in the tests): the noise, engine phases,
siren phases and horns come from generators seeded by (seed, actor id, purpose).

## Validation

Measured 2026-10-30 on the reference machine (AMD Ryzen 9 9900X, 24 threads; numpy 2.3, Python 3.14), 20 °C, 70 % RH.

**(a) A source passing at a known speed** (`falcon derive audio-check passby`, also in the tests): a pure tone of
100 dB at 1 m on a straight line at constant speed past one stationary microphone (no ground), over the stretch within
100 m. The measured frequency is the analytic signal's phase slope (10 ms); the theory is f0·c/(c − v_r) with v_r the
source's speed towards the microphone at its emission time, from the closed-form solution of (x0 + v·τ)² + d² =
c²·(t − τ)² — independent of the implementation's Newton solver. The level's theory is 100 − 20·lg r − α(f0)·r.

| pass-by | heard frequency span | frequency error, max / median | level error, max / median |
|---|---|---|---|
| 1 kHz, 20 m/s, 10 m off the line | 945.5–1061.3 Hz | 0.0035 / 0.0005 Hz | 0.015 / 0.004 dB |
| 2 kHz, 30 m/s, 3 m off | 1839.3–2191.5 Hz | 0.065 / 0.002 Hz (at the closest point, where the frequency falls fastest) | 0.087 / 0.002 dB |
| 8 kHz, 15 m/s, 40 m off (α = 0.078 dB/m: up to 7.7 dB absorbed) | 7738.0–8294.1 Hz | 0.62 / 0.26 Hz (≤ 0.008 %: the cubic read's phase error at 8 kHz) | 0.17 / 0.12 dB |

Also checked in the tests: ISO 9613-1 against ISO 9613-2's Table 2 (four climates, eight bands, within its rounding),
1/r at 5, 20 and 80 m (± 0.02 dB), the ground image's two-path sum (± 0.03 dB), the barrier formula behind a bus
(± 0.6 dB), the retarded-time equation on a curving source (< 1 µs), the CNOSSOS band powers of an emitted engine
(± 0.5 dB per band), the siren's 123 dB(A) at 3 m and the horns' levels (± 0.5 dB), and bit-identical output for the
same seed.

**(b) Direction of arrival on a real recorded drive.** A 60 s drive in central Bengaluru (world-8c0095b7, left-hand,
the Indian profile with fleet weights 0.04 ambulance, 0.03 fire engine and 0.04 police car added, 60 vehicles and 40
walkers, seed 7, 20 steps/s, CARLA 0.9.16 on Falcon's server), the default 8-microphone array (radius 0.2 m, 1.55 m up).
Derived: 22 engines, 3 sirens (two police cars on yelp, one ambulance on wail/yelp), 30 horn events on 13 vehicles
(pedestrian ahead 19, slow leader 9, queue 2; 21 by two-wheelers), a median of 12 audible sources per step (up to 18),
the loudest at a median 96 dB, Doppler factors 0.93–1.10. The estimator is SRP-PHAT (DiBiase et al. 2001; numpy, in
`core/audio.py`): 2048-sample Hann frames centred on every step, PHAT-weighted cross-spectra of all 28 pairs in
200–6000 Hz, steered over 1° of azimuth × elevations −10…20°, far field. Scored against the ground truth's loudest
source on steps where it is ≥ 3 dB louder than the next:

| drive variant | steps with a dominant source | azimuth error, median / 90th percentile | by the dominant source's kind |
|---|---|---|---|
| as derived (sirens on) | 1129 of 1198 | **0.62° / 167°** | siren 1054 steps: 0.77° / 169°; engine 72: 0.35° / 33°; horn 3: 0.25° / 0.46° |
| sirens off (same drive, same seed) | 795 of 1198 | **0.13° / 0.68°** | engine 741: 0.11° / 0.62°; horn 54: 0.43° / 16° |

- With sirens on, the loudest source is a siren 100 m away (median) on 88 % of the steps. On the 509 of those steps
  where SRP-PHAT misses it by more than 10°, it points **within 3° of another audible source every time** — a bus
  (358), a two-wheeler (108) or a car (23) engine at a median 18 m, 13.5 dB (median) quieter. That is PHAT's known
  behaviour, not a ground-truth error: the weighting whitens every bin, and a siren's few harmonic lines lose to a
  nearby broadband engine that fills every bin. A single siren alone is found to the degree (checked separately: alone at
  20, 100 and 190 m, every frame within 1°). A siren detector should not rely on broadband PHAT.
- Against the nearest audible source (any of them), the error is 0.19° median, 0.94° p90 — but a random azimuth already
  scores 16° / 55° there with 12 sources around, so that number says little.
- These errors are an **upper bound on real performance**: free field, no reverberation, no ego-body shadowing or
  reflection, no wind; only the microphones' 29 dB self-noise. The 1° grid limits the median.

**(c) Cost**: 125 s of compute for the 60 s drive with sirens (38 sources: 22 engines, 3 sirens, 13 horn segments) —
**125 s per minute of audio**: 7 s reading the run and planning the horns, 33 s synthesising the emissions, 84 s
propagating to 8 microphones and their ground images, plus 23 s for the SRP-PHAT check (included). 105 s without the
sirens (35 sources, 0.85 GB peak memory); 92 MB of WAV per minute (8 × float32 × 48 kHz). The cost grows with the
number of sources within `max_range_m` and the number of microphones; one process, numpy, no GPU.

## Limits
- **Synthetic sources**: emission models (CNOSSOS-EU is a statistical average over a European fleet, made for noise
  mapping) and heuristics (engine orders, horns, siren sweeps), not recordings of real vehicles — and no measured
  Indian vehicle data. Spectral detail inside an octave band is invented; tyre and engine tones of real vehicles are
  richer. A detector trained on it learns what these models sound like.
- **Not full acoustics**: free field plus one ground reflection; no reverberation, no building reflections or
  shielding, no ego-body effects, no wind, no refraction, no directivity. Street canyons in a city change levels and
  directions far more than this model says.
- **Horns are a rule set**, not observed honking; their rates are Falcon's guesses for an Indian street.
- **Sirens follow a US specification**; the level is the specification's minimum on axis, radiated all round.
- **Occlusion** only by vehicles' boxes; the diffraction formula is for a thin barrier.
- **Paths** come from the recorded poses at the simulation rate (20 Hz by default), interpolated; vibrations or
  motion faster than that are not in the audio.
