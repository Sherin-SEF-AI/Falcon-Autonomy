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

## Limits

- Appearance only (and object statistics when labelled): no lidar or radar gap yet.
- The feature space is one ImageNet network's; another network would give other numbers. The floor makes the scale
  readable, not universal.
- Matching is global per set; it cannot add what CARLA does not render (lens artefacts, sensor-specific colour
  processing, real textures).
