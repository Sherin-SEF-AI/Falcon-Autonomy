# Weather and sensor physics

CARLA 0.9.16's lidar and radar ignore weather, and its camera shows rain only as water on the lens and wet surfaces
([CARLA lessons](carla-lessons.md)). Falcon adds what a real sensor would suffer, as seeded post-processing of the data
the workers record and show (`falcon_autonomy/core/sensorphysics.py`; numpy and OpenCV, which CARLA's Python has).
Each model is taken from a published relation where one exists; every other number is labelled **heuristic** here and
in the code. These are models, not measurements of a real sensor: what each can and cannot say is under *Limits*.

## Turning it on — never silently

Nothing is applied unless the run's config carries an enabled block (a world in the project may carry the same block,
for the drive config to copy):

```json
"sensor_physics": {"enabled": true, "lidar": true, "radar": true, "camera": true, "rain_mm_h": null,
                   "lidar_params": {}, "radar_params": {}, "camera_params": {}}
```

- `lidar`, `radar`, `camera`: each model on or off.
- `rain_mm_h`: the rain rate; `null` reads it from CARLA's precipitation (below).
- `*_params`: any parameter of the tables below, by name. An unknown key refuses the run.
- A rig sensor opts out with `"weather_physics": false`.

What ran is recorded: `run.json` → `sensor_physics` holds the conditions, every parameter, the models with their
sources, the order, and per sensor how many frames each effect touched (`{"enabled": false}` when off); each frame an
effect touched lists it in `frames.jsonl` → `"physics": {"<sensor>": ["droplets", "glare", …]}`, as faults are
listed under `"faults"` (an effect listed is a model applied to the frame: `glare` adds nothing where nothing is
bright). Effects: `droplets`, `wet_contrast`, `glare`, `headlight_glare` (lamps were projected into the
view; a lamp found hidden by the depth image is then left out), `lidar_attenuation`, `lidar_dropout`, `lidar_clutter`,
`radar_dropout`, `radar_clutter`.

Where it applies: recorded drives (`falcon drive`, the data engine, scenario runs) — cameras, lidars and radars; Live
— the view when it is a rig camera (not the chase or free view; no headlight glare there) and the recordings' cameras;
a live weather event changes the conditions. Stack testing, HIL and the gym environment do not apply it yet: what they
publish is CARLA's data as it is.

## Conditions

| Quantity | From | Source |
|---|---|---|
| Visibility V | 7000 m / `fog_density` | measured on CARLA 0.9.16 at a 5 % contrast threshold ([CARLA lessons](carla-lessons.md)) |
| Visible extinction | −ln 0.05 / V = 3.0 / V | the same 5 % threshold (Koschmieder's law) |
| Rain rate R | 50 mm/h × `precipitation` / 100, unless `rain_mm_h` | **heuristic**: CARLA's precipitation has no physical unit; 50 mm/h is where "violent" rain begins |
| Lens wetness | `precipitation` / 100 | **heuristic** |
| Night | `sun_altitude_angle` < 0 | — |

## Lidar

Applied to each slice in the sensor's own frame before the sweep is assembled (so each point's range is from the
sensor that cast it), after the label proof and the vehicle-box check (which judge CARLA's own data).

**Extinction** α = α_fog + α_rain (1/m).

- Fog: α(λ) = (3.0 / V)·(λ / 550 nm)^−q, with q = 1.6 (V > 50 km), 1.3 (6–50 km), 0.16 V + 0.34 (1–6 km), V − 0.5
  (0.5–1 km), 0 (< 0.5 km), V in km — I. I. Kim, B. McArthur, E. Korevaar, *Comparison of laser beam propagation at
  785 nm and 1550 nm in fog and haze for optical wireless communications*, Proc. SPIE 4214 (2001). Their 3.91 is for a
  2 % threshold; Falcon keeps 3.0 because CARLA's visibility relation was measured at 5 %.
- Rain: A = 1.076 R^0.67 dB/km — Carbonneau's empirical relation for free-space optics (France, a temperate climate),
  as tabulated in FSO rain-attenuation studies. A 2009–2012 Czech FSO link measured attenuation in agreement with it;
  measurements in tropical Malaysia did not (they fitted 0.4195 R^0.8486). Rain drops are much larger than the
  wavelength, so the near-infrared value applies at 905 nm.

**Dropout.** A return CARLA gave is a clear-air return. It survives the weather with probability

  p = exp(−(e^{2αd} − 1)·d² / (κ ρ' R_max²))

the ratio of the detection probabilities with and without the two-way transmission e^{−2αd}, for a return whose
power has fully developed speckle (exponentially distributed: J. W. Goodman, *Speckle Phenomena in Optics*, 2007) and
whose mean over the threshold is κ ρ' (R_max / d)²: R_max is the rig's `range`, ρ' the surface's reflectivity over
`rho_ref`, κ = 1 / ln 2 (a reference target at the rated range is detected half the time).

**Intensity.** CARLA's intensity × e^{−2αd}.

**Backscatter clutter.** The medium in front of a target returns light too.

- Fog: a range cell of length Δr = cτ/2 at range r acts as a target of reflectivity π β Δr, with β = α_fog / S and the
  lidar ratio of water droplets S = 18.8 sr at 905 nm (E. J. O'Connor, A. J. Illingworth, R. J. Hogan, *A technique
  for autocalibration of cloud lidar*, J. Atmos. Oceanic Technol. 21(5), 777–786, 2004), seen through the overlap of
  transmitter and receiver O(r) = min(1, (r / r_overlap)²).
- Rain: discrete drops of the Marshall–Palmer distribution N(D) = 8000 e^{−ΛD} m⁻³ mm⁻¹, Λ = 4.1 R^−0.21 mm⁻¹
  (J. S. Marshall, W. McK. Palmer, *The distribution of raindrops with size*, J. Meteorol. 5, 1948); each drop a
  specular sphere (Fresnel reflectance of water 0.02, i.e. a Lambertian-equivalent 0.02 / 4) filling (D / w(r))² of a
  beam of width w(r) = w0 + θ r.
- Each cell is detected with the same speckle statistics; the cumulative probability of a weather return before each
  range is tabulated once per run and weather. The lidar reports the **strongest** return: a weather return in front
  replaces the target's with probability S_c / (S_c + S_t) (the chance one exponential signal exceeds the other) or
  stands alone where the target's return was lost.
- The structure follows M. Hahner et al., *Fog simulation on real LiDAR point clouds for 3D object detection in adverse
  weather*, ICCV 2021, and V. Kilic et al., *Lidar light scattering augmentation (LISA)*, arXiv:2107.07004 (2021).

**Labels.** A weather return is tagged **30, `weather_noise`** with actor id 0 (`.labels.bin`); its nuScenes-lidarseg
category is `noise` (index 0) and it has no panoptic instance. Real returns keep CARLA's tag and actor id and their own
position. The class is in `dataset.json`'s class table (`groundtruth.CLASSES`) and the lidar encoding there says so;
the MCAP/ROS exporter writes the tag as it is (30).

| Parameter | Default | Unit | Source |
|---|---|---|---|
| `wavelength_nm` | 905 | nm | typical automotive lidar |
| `rho_ref` | 0.1 | — | the rig's range read as rated for a 10 % target (a common datasheet basis) |
| `kappa` | 1 / ln 2 | — | a reference target at the rated range is detected half the time (**heuristic** choice) |
| `pulse_ns` | 3 | ns | **heuristic** (range cell 0.45 m) |
| `r_overlap_m` | 5 | m | **heuristic** near-field overlap |
| `r_min_m` | 0.5 | m | **heuristic** blanking |
| `lidar_ratio_sr` | 18.8 | sr | O'Connor et al. 2004 (water droplets, 905 nm) |
| `beam_w0_m`, `beam_div_rad` | 0.01, 0.003 | m, rad | **heuristic** typical |
| `drop_rho` | 0.005 | — | Fresnel reflectance of water / 4 |
| `backscatter` | true | — | clutter on or off |
| reflectivity per CARLA tag | road 0.12, car 0.4, vegetation 0.5, sign 0.8, … (`REFLECTIVITY`) | — | **heuristic**: typical 905 nm values, rounded |

## Radar

Applied to each measurement (CARLA's detections: velocity, azimuth, altitude, depth).

- **Rain attenuation**: γ = k R^α dB/km, Recommendation ITU-R P.838-3 (03/2005), table 5 at 77 GHz (kH 1.1320,
  αH 0.7177, kV 1.1276, αV 0.7073), combined by its equations (4)–(5) for a horizontal path and an unknown (45°)
  polarisation: k = 1.1298, α = 0.7125; two-way over each detection's range. 7.8 dB/km at 15 mm/h, 18.4 at 50 mm/h.
- **Detection**: Swerling I, P_d = P_fa^{1/(1+SNR)} (M. A. Richards, *Fundamentals of Radar Signal Processing*, 2014),
  SNR = SNR_ref (R_max / d)⁴ for a reference cross-section (CARLA's radar reports none), SNR_ref set so that P_d = 0.5
  at the rig's range (18.9 for P_fa = 10⁻⁶). In rain the signal is attenuated and the noise floor raised by the rain in
  the cell, SNR / (1 + CNR); a detection survives with the ratio of the two P_d.
- **Rain clutter**: volume reflectivity η = π⁵ |K|² Z / λ⁴ with Z = 200 R^1.6 mm⁶ m⁻³ (Marshall–Palmer), in the
  Rayleigh approximation — an **upper bound** at 77 GHz, where drops above about 1 mm scatter less than Rayleigh's law
  says — over the resolution cell r² θ_az θ_el Δr. Each cell where clutter exceeds noise (CNR > 1) gives a spurious
  detection with probability `p_spurious` (**heuristic**: what a CFAR detector leaves of rain clutter is not a
  published constant), at the radial velocity of drops falling at v(D0) = 9.65 − 10.3 e^{−0.6 D0} m/s (D. Atlas,
  R. C. Srivastava, R. S. Sekhon, Rev. Geophys. 11(1), 1973; D0 = 3.67 / Λ) seen from the moving radar.
- **Labels**: spurious detections are flagged in `radar/<sensor>/<step>.flags.npy` (uint8, 1 = weather clutter, in the
  order of the `.npy` rows), written only for frames the physics touched; `frames.jsonl` lists the file under
  `"<sensor>/radar_flags"`.

| Parameter | Default | Unit | Source |
|---|---|---|---|
| `frequency_ghz` | 77 | GHz | P.838-3 coefficients are tabulated here at 77 and 80 GHz |
| `p_fa` | 10⁻⁶ | — | **heuristic** design value |
| `pd_at_range` | 0.5 | — | the rig's range read as the 50 % detection range |
| `k_water` | 0.93 | — | |K|² of water at centimetre wavelengths (lower at 77 GHz: clutter an upper bound) |
| `sigma_ref_m2` | 10 | m² | **heuristic** (a car) |
| `range_res_m`, `az_res_deg`, `el_res_deg` | 0.5, 2, 5 | m, ° | **heuristic** resolution cell |
| `p_spurious` | 0.01 | — | **heuristic** |

The model's own numbers (P.838-3 attenuation, Rayleigh clutter): a reference target at 90 m of a 100 m radar is
kept with probability 0.64 at 15 mm/h and 0.12 at 50 mm/h; at 10 m, 0.996 and 0.97. Most of the far loss is the
clutter's desensitisation, which rests on the Rayleigh upper bound: real 77 GHz radars lose less.

## Camera

Applied to the image (a fisheye's after its remap) in the writer thread; the droplet layer is advanced in the run's
own thread in step order, so a seed always gives the same drops.

- **Lens droplets** (`droplets`): drops land at a rate that gives a steady-state cover of
  `cover_max` · R / (R + `rain_half_mm_h`) of the image and leave after an exponentially distributed residence time;
  radii are log-normal. Each drop is a small lens that shows an inverted, minified (`field` ×) and defocused patch of
  the scene around it, with a darker rim (total internal reflection) and a defocused outline — the geometry of
  adherent drops in M. Roser, A. Geiger, *Video-based raindrop detection for improved image registration*, ICCV
  Workshops 2009, and S. You et al., *Adherent raindrop modeling, detection and removal in video*, IEEE TPAMI 38(9),
  2016, simplified to a local displacement. The lens starts the run already wet (the layer is run to its steady
  state before the first frame).
- **Wet-lens contrast** (`wet_contrast`): a water film scatters a share s = `veil_max` × lens wetness into a veil,
  I' = (1 − s) I + s Ī (Ī: the image blurred by 5 % of its width) — **heuristic** magnitude.
- **Glare and bloom** (`glare`): bright sources spread by the lens with a point-spread function falling as 1/θ² (the
  form of the Stiles–Holladay disability-glare relation, CIE 146:2002 / J. J. Vos 2003), normalised over 20° and
  scaled by a veiling-glare index (ISO 9358) of 3 % on a dry lens, raised (×(1 + 4 · wetness)) on a wet one. Sources:
  clipped pixels (the sun, lamps) weighted by an over-exposure factor of 8 (their true radiance is clipped — the sun is
  far brighter, so its glare is underestimated).
- **Night headlights** (`headlight_glare`, pinhole cameras in drives): when the sun is below the horizon, the
  headlights of every vehicle within 150 m whose front faces the camera — **heuristic** positions: CARLA does not
  expose its lamps, so the front face of the actor's box, 0.65 m above its bottom and 0.3 m in from each side (one
  lamp for a motorcycle); assumed on at night whatever CARLA rendered; brightness: a lamp 10 m away adds 1000 8-bit
  levels 1° from itself on a dry lens (saturation within about 2°), falling with the inverse square of distance. A lamp
  behind a nearer surface of the camera's depth image is left out (cameras that record depth only).

| Parameter | Default | Unit | Source |
|---|---|---|---|
| `cover_max`, `rain_half_mm_h` | 0.25, 10 | —, mm/h | **heuristic** |
| `residence_s` | 3 | s | **heuristic** |
| `radius_median`, `radius_sigma` | 0.025, 0.45 | share of image height, — | **heuristic** |
| `field`, `blur`, `rim_dark`, `edge` | 3.5, 0.25, 0.35, 0.25 | — | **heuristic** |
| `veil_max` | 0.25 | — | **heuristic** |
| `vgi`, `wet_glare_gain` | 0.03, 4 | — | ISO 9358 typical range 1–4 %; gain **heuristic** |
| `overexposure` | 8 | — | **heuristic** |
| `glare_theta0_deg`, `glare_reach_deg` | 0.3, 20 | ° | **heuristic** core and reach |
| `headlight_veil_1deg`, `headlight_max_m` | 1000, 150 | levels, m | **heuristic** |
| `droplets`, `contrast`, `glare`, `headlights` | true | — | each effect on or off |

## Order with faults

Weather physics first, then the sensor's [faults of known size](index.md#rigs): the scene's light crosses the
atmosphere (CARLA's fog, the lidar's extinction), then the water on the lens (droplets, contrast), then the optics
(glare), and only then the sensor's own faults in `core.faults`' order — dirt, the `rain` fault of known size, the
`glare` fault, rolling shutter, noise, compression. Mount faults move the sensor before any of this. A frame a `drop`
or `freeze` fault removed gets no physics. The two are separate and composable: the `rain` fault is a test of known
size, the droplets are the weather's, and a frame can carry both, each recorded apart (`"faults"`, `"physics"`).

## Measured

**The lidar model against its own prediction**, on real data: 20 sweeps of a real CARLA 0.9.16 drive (Town10HD,
32 channels, 300 000 points/s, range 80 m; 27 556 points a sweep), each run through the model at several weathers.
Share of real returns kept per range bin, measured (the model's expectation, `expected_real_kept`, in brackets):

| Weather | V | 0–10 m | 10–20 | 20–30 | 30–40 | 40–60 | 60–80 m | weather returns | object sweeps with ≥ 5 points |
|---|---|---|---|---|---|---|---|---|---|
| clear (fog 0) | — | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0 % | 127 |
| fog 10 | 700 m | 1.00 (1.00) | 0.98 (0.99) | 0.96 (0.96) | 0.92 (0.92) | 0.85 (0.85) | 0.69 (0.69) | 4.2 % | 106 |
| fog 20 | 350 m | 0.99 (0.99) | 0.96 (0.96) | 0.90 (0.90) | 0.80 (0.80) | 0.62 (0.62) | 0.32 (0.32) | 10.1 % | 89 |
| fog 50 | 140 m | 0.97 (0.97) | 0.88 (0.88) | 0.67 (0.68) | 0.39 (0.39) | 0.14 (0.14) | 0.01 (0.01) | 24.2 % | 72 |
| fog 100 | 70 m | 0.93 (0.93) | 0.69 (0.69) | 0.28 (0.28) | 0.05 (0.05) | 0.00 (0.00) | 0.00 (0.00) | 38.5 % | 55 |
| rain 25 mm/h | — | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 0.99 (0.99) | 0.98 (0.98) | 0.94 (0.94) | 0.01 % | 126 |
| rain 50 mm/h | — | 1.00 (1.00) | 1.00 (1.00) | 0.99 (0.99) | 0.99 (0.99) | 0.97 (0.97) | 0.90 (0.90) | 0.04 % | 124 |

The implementation draws exactly its documented model (the tests check it bin by bin with a binomial tolerance at
four fog densities, with and without clutter). The last column is a **proxy detector**, not a model: the
vehicle and pedestrian actors with at least 5 points in a sweep, summed over the 20 sweeps — what a point-count
detector would lose. The camera models were not run through a detector: the harness's synthetic detector reads the
ground truth, not the image, so it cannot show an image effect, and no real detector was evaluated for this.

**Cost** (this machine, one 1280 × 720 frame): the camera models about 40–70 ms (droplets about 30 ms with 140 drops,
contrast and glare about 20 ms each, computed at reduced size because both are smooth), in the drive's writer threads;
the lidar 1.2 ms per 14 000-point slice.

## Limits

- The models are not calibrated against a real sensor in real weather: their **structure** is published physics,
  several **magnitudes** are heuristic (marked above). The fog clutter in particular depends on the overlap, pulse
  length and threshold of the real lidar, which CARLA's lidar does not define; it may be off by an order of magnitude.
  CARLA's presets' haze (fog_density 2–7) is treated as fog: with the droplets' lidar ratio its clutter is
  overestimated (haze has a lidar ratio nearer 50 sr).
- Lidar clutter is drawn only along rays that returned a point in CARLA: CARLA reports no ray that hit nothing, so rays
  into open sky get none (an underestimate where much of the field of view is sky). Multiple scattering, the drops'
  glory, the pulse's shape, wet-road specular loss and spray from other vehicles are not modelled.
- Carbonneau's rain relation is for a temperate climate; tropical (monsoon) rain attenuates differently.
- CARLA's radar reports no cross-section, so every detection is treated as the reference target; the rain clutter is
  a Rayleigh upper bound; the spurious-detection rate is heuristic.
- The camera's droplets are a geometric caricature (no ray tracing, no streaks from motion or wind, no wiper); the
  glare from the sun is underestimated (clipped radiance); headlights are assumed on at night and their positions are
  a heuristic, whatever CARLA drew. Fog on the camera is CARLA's own rendering; nothing is added to it.
- Rain on the lens and a dry road at the same time, or the reverse, are CARLA's weather fields as given; the physics
  reads precipitation as rain on the lens.
