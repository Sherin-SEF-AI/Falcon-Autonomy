# The sim-to-real gap

Models ▸ Sim-to-real gap (or `falcon gap`) measures how far a set of Falcon drives' camera images — and their labels —
are from a real dataset's, statistic by statistic, and in a fixed feature space.

## Real sets

Read in place; nothing is copied into the project.

| kind | what is read |
|---|---|
| `folder` | any folder of images (recursively; .png .jpg .jpeg), no labels |
| `kitti_object` | KITTI object: `image_2/*.png` and `label_2/*.txt` (Cyclist counts as bicycle, Van as car, Tram as train; Misc and DontCare left out; range to the box's bottom centre) |
| `kitti_raw` | a KITTI raw drive (or a date folder of drives): `image_02/data/*.png`, the left colour camera; tracklets are not read |
| `nuscenes` | a nuScenes release folder (`v1.0-mini`, `v1.0-trainval`, …): one camera's key frames (CAM_FRONT by default); boxes are the annotated 3D boxes projected (the corners in front of the camera, clipped to the image), looser than drawn 2D boxes; night and rain from the scene descriptions |
| `bdd100k` | an image folder and BDD100K's detection labels JSON (weather, time of day and scene from its attributes) |
| `coco` | an image folder and a COCO instances JSON (category names mapped to Falcon's classes; unmapped ones listed) |
| `falcon` | another Falcon run |
| `dashcam` | an image set extracted from your own dashcam (`falcon dashcam extract`, [dashcam](dashcam.md)): no labels; each frame's GPS position, speed, clip and sun altitude as its conditions |

`--images N` takes a seeded random subset of each side.

## Statistics

Every image is first resized to a common width of 640 px, so that resolution does not dominate. Per image:

| statistic | meaning |
|---|---|
| L_mean, L_std | CIELAB lightness: mean (brightness) and spread (contrast) |
| a_mean, b_mean, ab_std | CIELAB colour: green–red and blue–yellow means, chroma spread |
| saturation | HSV saturation, mean |
| dark_share, bright_share | crushed blacks (L < 2) and clipped highlights (a channel at 255) |
| noise_sigma | noise, 8-bit levels (Immerkær 1996; texture raises it a little) |
| sharpness | log10 of the Laplacian's variance **less what the noise gives it** (20σ² for this kernel) — so a noisy image does not read as sharp |
| spectrum_slope | slope of the radially averaged power spectrum, 0.02–0.4 cycles/px (natural images ≈ −2) |

Labelled sets add: class shares, objects per image, box height as a share of image height and range. Objects lower
than 1.5 % of the image height are left out on both sides (a share, not pixels, so different resolutions compare).

**Night statistics** (`--night on`, or by themselves when the real set is mostly at night — a dashcam set's frames
say so by the sun's altitude; heuristics, labelled as such). The statistics above stay as they are; these are added:

| statistic | meaning |
|---|---|
| dark_share_10, dark_share_25 | share of pixels at L < 10 (deep shadow) and at L < 25 (shadow) |
| light_share | share of pixels at L > 90: light sources and what they light |
| light_blobs | bright blobs (L > 90, at least 4 px at 640 px wide) per image: the light sources |
| halo_L | the glow round the light sources: mean L 3–8 px outside the bright blobs less the image's median L (bloom and lens glare raise it); none without a light |
| noise_dark | noise in the shadows: Immerkær's estimate on the pixels at L < 25 only (eroded by a pixel); none with fewer than 1000 such pixels |

Each statistic's gap is the **1-Wasserstein distance** between the two sets' distributions. Its **shift** — that
distance over the real set's interquartile range — ranks the gaps. The shift is a heuristic scale, not a test. Class
shares are compared by total-variation distance.

## Fréchet and kernel distances

FD and KD apply the method of FID and KID (Heusel et al. 2017; Bińkowski et al. 2018) to the 2048 pooled features of a
fixed network: torchvision's ResNet-50 with its ImageNet-1k weights (V2), the whole image resized to 224 × 224 with
no crop. The weights download only when asked (*Download the feature extractor…*, `falcon gap --fetch`):
97.8 MB from download.pytorch.org, SHA-256 `11ad3fa6…379ca` checked. torchvision is BSD-3-Clause; the
ImageNet-1k training data is under its own terms.

These numbers are **not comparable with published FID**, which uses a specific Inception network, and both depend on
the number of images. So each is reported next to its **floor**: the same distance between two random halves of the
real set, at the same set sizes. A gap near the floor is as close as two real samples are to each other. KD is
unbiased; its ± is the spread over subsets.

## Appearance matching

The optional *appearance matching* is a fixed, non-learned transform of the simulated images, with its effect
measured rather than assumed:

1. **Colour.** CIELAB histogram matching: one lookup table per channel from the simulated set's pooled histogram to
   the real set's, the same for every image, so images keep their differences.
2. **Blur.** A Gaussian blur fitted, after the colour step and with the noise of step 3, to the real set's median
   sharpness. When the real set is softer than the largest blur tried, no blur is applied, and the fit says why.
3. **Noise.** Gaussian noise added up to the real set's median noise level.

Learned translation, which changes local texture and rendering, is [sim-to-real translation](translation.md).

## Measured: CARLA against KITTI

Two Falcon drives on a real CARLA 0.9.16 server (Town10HD and the imported central Bengaluru, 1280 × 720, both
ClearNoon; 240 images in all) against 240 images of three KITTI raw drives (2011_09_26: 0005, 0015,
0027; left colour camera). `falcon gap --features --match`, seed 0, measured 2026-10-28:

| statistic | CARLA (median) | KITTI (median) | shift | after matching |
|---|---|---|---|---|
| L_mean | 62.9 | 38.8 | 4.56 | 38.6 (shift 2.82) |
| L_std | 14.9 | 29.0 | 3.84 | 24.1 (1.46) |
| spectrum_slope | −2.84 | −2.46 | 3.69 | −2.81 (3.92) |
| sharpness | 2.19 | 2.98 | 2.29 | 2.76 (1.67) |
| saturation | 0.14 | 0.27 | 2.15 | 0.44 (2.33) |
| ab_std | 5.0 | 11.2 | 1.64 | 9.2 (0.68) |
| noise_sigma | 1.45 | 2.68 | 0.75 | 3.59 (0.23) |

FD 184.8 against a real-vs-real floor of 23.1; KD 0.113 ± 0.009 against a floor of −0.001 (120 against 120 images).
After appearance matching: FD 195.3, KD 0.124.

**What this says.** CARLA's images are brighter, flatter, less saturated and softer than KITTI's, with less noise
and a steeper spectrum (less fine detail). Matching moves brightness, contrast, colour spread and noise most of the way. It does **not**
reduce the feature distance, which even rises a little: what separates the two sets in a network's features is
content and rendering — what is in the scenes, the textures, the geometry — not global colour. The
scenes also differ (Karlsruhe in 2011 against CARLA's town and Bengaluru), so part of the feature gap is the
places, not the renderer.

## At night

**CARLA at night against the user's night dashcam frames**: two night drives on a real CARLA 0.9.16 server
(ClearNight — CARLA's preset, sun −90°, fog density 60 —, 60 s at 10 frames/s, 640 × 360, every vehicle's position
lights and low beams on: [vehicle lights](protocol.md)), Town10HD (30 vehicles; the town's own street and building
lights: 208 of its 401 lights on) and the re-imported central Bengaluru (60 vehicles; an imported world has no
lights of its own), against the 600 held-out night frames of [the user's dashcam](dashcam.md#at-night) (sun below
−6°). `falcon gap --features`, 300 images a side, seed 0, measured 2026-10-29:

| statistic | CARLA (median) | dashcam (median) | shift |
|---|---|---|---|
| L_mean | 4.9 | 16.5 | 1.73 |
| L_std | 8.6 | 17.6 | 2.11 |
| dark_share (L < 2) | 0.47 | 0.02 | 6.79 |
| noise_sigma | 2.78 | 1.32 | 2.13 |
| spectrum_slope | −1.70 | −2.46 | 3.56 |
| sharpness | 2.17 | 2.70 | 1.32 |
| *noise_dark* | 2.80 | 0.54 | 14.8 |
| *halo_L* | 8.6 | 29.6 | 2.07 |
| *light_blobs* | 0 | 36 | 1.72 |
| *dark_share_25* | 0.94 | 0.80 | 1.43 |

FD **181.3** against a real-vs-real floor of 29.8; KD 0.117 ± 0.004 against −0.001 (150 against 150 images).
For scale, the dashcam's own night frames against its own day frames (the same camera, the same region): FD 129.0
(floor 31.2), KD 0.101.

**What this says.** CARLA's night is far darker than the camera's (half of its pixels are black; the camera's
auto-exposure lifts the scene to a median L of 16), and its shadows carry rendering noise where the camera's are
smoothed by its night processing (noise in the shadows 2.80 against 0.54 levels). The camera sees dozens of light
sources with wide halos per frame; the median simulated frame has none (the imported Bengaluru has no street lights
at all, so only the cars' own lights show; Town10HD's lamps are fewer and sharper). Its frames are blurrier
(motion at long exposure) and their spectrum steeper; the simulated frames' flat −1.7 slope is their noise.

## Limits

- Appearance only (and object statistics when labelled): no lidar or radar gap yet.
- The night statistics' thresholds (L 10, 25, 90; the halo's 3–8 px ring) are fixed choices, not fitted; one
  camera's night footage measured.
- The feature space is one ImageNet network's; another network would give other numbers. The floor makes the scale
  readable, not universal.
- Matching is global per set; it cannot add what CARLA does not render (lens artefacts, sensor-specific colour
  processing, real textures).
