# Neural reconstruction of real logs

A recorded drive — its posed camera images, and its lidar when it has one — becomes a scene of 3D Gaussians that can
be rendered from poses the drive never took: the ego moved sideways, another camera mount, another field of view. It
complements [log replay](log-replay.md): a replay rebuilds a log's trajectories with CARLA's meshes; a reconstruction
keeps the log's own pixels — of the **static scene** only. The code is `core/recon/`; the page is a section of
*Replay*.

```sh
falcon recon train --source falcon:~/demo.falcon/runs/drive --out ~/demo.falcon/recon/drive
falcon recon train --source kitti_raw:~/kitti/2011_09_26/2011_09_26_drive_0005_sync --out ~/demo.falcon/recon/k5
falcon recon train --source 'nuscenes:~/data/nuscenes?scene=scene-0061' --out ~/demo.falcon/recon/s61
falcon recon eval ~/demo.falcon/recon/drive --warp-shift 0.5 --warp-shift 1.0 --warp-check 1.0 \
    --reference falcon:~/demo.falcon/runs/other-lane
falcon recon render ~/demo.falcon/recon/drive --frame 40 --lateral 1.0 --out view.png --depth-out range.png
falcon recon rerender ~/demo.falcon/recon/drive --lateral 1.0 --out ~/demo.falcon/runs/drive-shifted-1m
falcon recon info ~/demo.falcon/recon/drive
```

`train` also takes `--iterations` (7000), `--width` (640: images are worked on at most this wide), `--init
auto|lidar|sfm|range|random`, `--holdout` (8), `--seed`, `--sh-degree` (3), `--max-gaussians` (600 000), `--every`,
`--camera`, `--device`, `--no-masks`, `--no-semantic`, `--config FILE`. `render` and `rerender` take `--lateral`
(metres to the right), `--forward`, `--up`, `--yaw` (degrees, CARLA's sense: positive turns right), `--mount
x,y,z,roll,pitch,yaw` (the camera's mount in CARLA's vehicle frame, as in a rig), `--fov` and `--size WxH`. Exit codes
are the CLI's ([cli](cli.md#exit-codes)).

## The method

3D Gaussian Splatting (Kerbl, Kopanas, Leimkühler and Drettakis, *3D Gaussian Splatting for Real-Time Radiance
Field Rendering*, SIGGRAPH 2023): the scene is a set of anisotropic 3D Gaussians — a mean, a scale and a rotation
(Σ = R S Sᵀ Rᵀ), an opacity and a colour as spherical harmonics up to degree 3 — projected to the image through the
local affine approximation of the perspective map (EWA splatting: Zwicker, Pfister, van Baar and Gross, *EWA
Splatting*, IEEE TVCG 2002), sorted by depth and alpha-composited front to back; their parameters are optimised
against the images by gradient descent through that differentiable rasterisation, while *adaptive density control*
clones, splits and prunes them. Falcon's implementation is written from the papers. **The Inria reference code is not
used** (its licence is non-commercial).

## The rasteriser: pure PyTorch

This machine has torch 2.9.1 + CUDA 12.8 and Python 3.14 but no CUDA toolkit (no `nvcc`), so gsplat's CUDA kernels
cannot be compiled here, and its prebuilt wheels do not fit: on 2026-10-29 its wheel index offered gsplat 1.5.3 for
CPython 3.10 only, torch 2.0–2.4 and CUDA 11.8–12.4 (PyPI has only the source-compiling wheel). No Python 3.10 is
installed, and torch ≤ 2.4 with CUDA ≤ 12.4 does not run on an RTX 5080 (Blackwell needs CUDA 12.8), so a
project-local venv would not have helped without system changes. Falcon therefore has its own differentiable
rasteriser in plain PyTorch (`core/recon/raster.py`), which runs wherever torch runs — any CUDA GPU torch supports, or
the CPU — in float32 or float64.

How it works:

- **Projection** of every Gaussian (vectorised): the camera-frame mean, the 2 × 2 screen covariance Σ' = J W Σ Wᵀ Jᵀ +
  0.3 I (the paper's dilation; x/z and y/z clamped to 15 % beyond the field for the Jacobian, as the paper does) and
  its inverse.
- **Exact bounds**: a pixel gets α = min(0.99, o·exp(−½ dᵀ Σ'⁻¹ d)) and α < 1/255 is skipped (as in the paper), so a
  Gaussian touches exactly the pixels inside dᵀ Σ'⁻¹ d ≤ 2 ln(255 o) — its axis-aligned box is used to bin it,
  tighter than the paper's 3σ circle and losing nothing.
- **Tiles**: the depth-sorted Gaussians are listed per 8 × 8 pixel tile (a stable sort by tile keeps the depth order).
- **Chunks**: tiles with similar list lengths are composited together, each list padded to the chunk's longest, as one
  tensor of (tiles × 64 pixels × Gaussians) — capped at 2²⁴ elements on a GPU — so memory is bounded whatever the
  scene; every chunk runs under activation checkpointing, so the backward pass recomputes it rather than storing it.
- **Compositing**: Tᵢ = Πⱼ<ᵢ(1 − αⱼ), wᵢ = αᵢTᵢ, colour Σ wᵢcᵢ, expected depth Σ wᵢzᵢ, alpha 1 − T_N, and on request
  the **median depth** — the z of the Gaussian at which T falls below ½, with that Gaussian's own flags (is it the
  sky?). Unlike the CUDA kernels there is no early stop at T < 10⁻⁴ (a difference below 10⁻⁴ of a colour).
- **Gradients** are torch's autograd through these operations; `tests/test_recon.py` compares every one (means,
  scales, rotations, opacities, colours, spherical harmonics) with central finite differences in float64.

**Speed** (measured 2026-10-29, RTX 5080, torch 2.9.1 + CUDA 12.8; CPU an AMD Ryzen 9 9900X, 12 threads): one
640 × 360 view of random Gaussians spread over 2–62 m in front of the camera, with degree-3 spherical harmonics,
forward and backward:

| Gaussians | tile | (Gaussian, tile) pairs | forward | backward | peak GPU memory |
|---|---|---|---|---|---|
| 100 000 | 8 px | 0.45 M | 13 ms | 34 ms | 1.1 GiB |
| 100 000 | 16 px | 0.22 M | 17 ms | 49 ms | 1.1 GiB |
| 300 000 | 8 px | 1.3 M | 25 ms | 85 ms | 1.4 GiB |
| 300 000 | 16 px | 0.65 M | 45 ms | 147 ms | 1.3 GiB |
| 600 000 | 8 px | 2.7 M | 50 ms | 172 ms | 1.7 GiB |
| 600 000 | 16 px | 1.3 M | 88 ms | 293 ms | 1.6 GiB |
| 300 000, on the CPU | 8 px | 1.3 M | 0.47 s | 2.26 s | — |

So 8-pixel tiles are the default. In real training (below) a step — render, loss, backward, Adam, density control —
ran at 8.9–11.1 steps a second with 270 000–600 000 Gaussians at 640 × 360 — a 7000-step reconstruction in 11–13
minutes — and a render for viewing takes 26 ms (440 000 Gaussians, 640 × 360, the median of 19 poses). That is slower
than CUDA kernels: the paper reports real-time rendering (at least 30 frames a second at 1080p) and training in
minutes (about 40 for 30 000 steps) on an A6000; gsplat could not be run here to compare on the same machine, so no
same-machine ratio is claimed. On the CPU a step at this size takes about 3 s: CPU training is for tests and small
images.

## Sources

Read in place; nothing is copied into the project (KITTI's licence, CC BY-NC-SA 3.0, forbids redistributing it
anyway; the tests write their own KITTI- and nuScenes-shaped folders with made-up numbers).

| source | images and poses | moving objects masked | lidar for the start |
|---|---|---|---|
| `falcon:RUN` | a pinhole camera's `rgb/`; the ego pose of `frames.jsonl` and the camera's actual mount (its faults included); K from its field of view | every vehicle and walker (instance labels → objects with a CARLA actor id; without instance labels, the vehicle and person classes — parked map cars too) and the ego's body | every sweep's static returns (actor id 0, not the ego, not weather noise) through `lidar_pose` |
| `kitti_raw:DRIVE` | `image_02/` (or `image_03/`), rectified; OXTS poses exactly as log replay reads them; the date folder's calibration, cam ← world = [I \| K⁻¹·P_rect's offset] · R_rect_00 · T_cam0←velo · T_velo←imu · T_imu←world (the devkit's chain) | **none**: the tracklets are a separate download and are not read, so moving vehicles stay in the images and blur the reconstruction where they pass | `velodyne_points/` (not motion-compensated: KITTI raw's own limit) |
| `nuscenes:ROOT?scene=NAME` | one camera's key frames (`CAM_FRONT` by default), each with its own ego pose and calibrated sensor | the key frame's annotated boxes of Falcon's actor classes, projected (edges clipped at the near plane), filled and dilated 2 px | `LIDAR_TOP` key-frame sweeps, points inside annotated boxes left out |

Checked: a Falcon run's lidar points against its own range labels (median |difference| 0.12 m for every sweep's
points projected into one frame, occluded points included); KITTI's chain against the devkit's own formula in the
tests (to 10⁻⁶ px) and by eye on drive 0005 (velodyne points on poles, a cyclist and a van); nuScenes on synthetic
tables only — **no nuScenes data is on this machine**, so it is supported by the format and not verified on real files.
ROS 2 bags and Waymo logs are read for trajectories only (their images are not), so they cannot be reconstructed.

**Frames.** The reconstruction's world is the source's own, right-handed, z up: a Falcon run's CARLA world mirrored (x,
−y, z), KITTI's local east-north-up about the first OXTS fix (log replay's projection), nuScenes' global frame. CARLA's
convention of any of them is the same mirror, as in log replay, so a shifted ego path is written into a falcon-raw run
as log replay would write it. Cameras are OpenCV's optical frame; intrinsics are in Falcon's pixel-edge convention
(an OpenCV principal point + ½). The `.ply` is in the reconstruction's world frame.

## Starting points

`--init auto` uses the source's lidar when it has one, else **structure from the known poses**: SIFT features (Lowe
2004) matched between frames about 1 m apart (Lowe's ratio test, 0.75), triangulated with the known projection
matrices, kept when both reprojection errors are below 1.5 px, the rays meet at over 1° and the depth is 0.5–200 m
(heuristic thresholds), masked pixels never used. `range` (Falcon runs only) samples the ground-truth range labels — a
simulator's privilege, never the default — and `random` fills the cameras' surroundings. Points are averaged per
voxel (0.15 m; 0.05 m for structure from poses), at most 300 000, coloured from the frames that see them. Every start
adds a **far shell** of 20 000 Gaussians marked *sky*, at 1.5 × the 99th percentile of the points' distance plus the
path's radius (at least 100 m), from 10° below the horizon to the zenith, so the sky and the far background have
something to become (heuristic); density control leaves the shell alone.

## Training

The paper's recipe at a shorter schedule: one image per step in a seeded random order; loss 0.8·L1 + 0.2·(1 − SSIM)
over the unmasked pixels; Adam with the paper's learning rates (positions 1.6·10⁻⁴ × the scene's extent decaying
exponentially to 1.6·10⁻⁶ ×, SH DC 2.5·10⁻³ and the rest 1/20 of it, opacity 0.05, log-scales 5·10⁻³, rotations
10⁻³; the extent is 1.1 × the largest distance of a training camera from their mean); spherical harmonics one degree
more every 1000 steps; density control every 100 steps from step 500 to half the run (clone below 1 % of the extent,
split above, gradient threshold 2·10⁻⁴ in the paper's units, prune below opacity 0.005 and, after the first opacity
reset, above 20 px on screen or 10 % of the extent), opacities reset every 3000 steps. **7000 steps by default** (the
paper trains 30 000 and reports its 7000-step results as somewhat lower); a cap of 600 000 Gaussians (the largest
gradients densified first) bounds memory — the paper has none. Every setting is in run.json.

Held out: every 8th frame (0, 8, 16, …: the Mip-NeRF 360 convention the paper follows) is never trained on.
**Semantic classes**: when the source has semantic labels (Falcon runs), each Gaussian's classes are voted after
training — its rendering weights summed per class over the training pixels (computed exactly by one backward pass per
view) — so re-rendered views carry an estimate of the semantic labels.

A CUDA training is a `needs_gpu` job: from the GUI the queue holds the GPU lock; from the CLI `falcon recon` takes it
itself and waits while a CARLA job or another GPU job holds it. Its share of GPU memory is capped at 60 % and its peak
measured. Seeded throughout: on the CPU with one thread two runs give the same Gaussians bit for bit (tested); with
several CPU threads the backward pass accumulates in a varying order, so runs differ in the last bits; on CUDA two
runs were identical (measured below).

**Output** (`--out`): `point_cloud.ply` (+ `extras.npz`: the sky flags and class votes), `views.json` (every frame's
pose, intrinsics and split), `curves.jsonl` (loss, PSNR of the step's image, the number of Gaussians, held-out PSNR
every 1000 steps), `evaluation.json`, `renders/` (a few held-out frames beside their renders), `run.json`
(provenance: the source's files hashed, settings, seed, versions, device, timings, peak GPU memory). About 110 MB for
440 000 Gaussians (the `.ply` is most of it).

**The `.ply`** is the layout every 3DGS viewer reads: binary little-endian, one `vertex` element of floats `x y z nx ny
nz f_dc_0..2 f_rest_0..44 opacity scale_0..2 rot_0..3` — `f_rest` channel-major (all of red's coefficients, then
green's, then blue's), opacity a logit, scales logarithms, `rot` a (w, x, y, z) quaternion, normals zero.

## The render API

```python
from falcon_autonomy.core.recon.scene import Recon
rec = Recon.load('~/demo.falcon/recon/drive')
cam = rec.camera(rec.frame(40), lateral_m=1.0)                 # or forward_m, up_m, yaw_deg, mount=, fov=, size=
cam = rec.camera_at({'x': 52.0, 'y': -61.5, 'z': 1.7, 'roll': 0, 'pitch': -5, 'yaw': 10}, fov=90)   # any pose
out = rec.render(cam)          # image (H, W, 3) uint8, range (H, W) m, alpha, sky, semantic (voted classes)
```

A camera is a world → optical matrix, K and a size; `rec.camera()` makes one from a frame of the drive with the ego
moved in its own frame, `rec.camera_at()` from a camera's pose in CARLA's convention of the reconstruction's world
(for a Falcon run: CARLA's own world coordinates). **Range** is the median depth's — the depth of the Gaussian at
which a pixel's transmittance falls below ½ — along the pixel's ray; 0 where it never falls that far or where that
Gaussian is on the far shell (falcon-raw's "nothing"). The expected depth Σwᵢzᵢ/α is not used: it mixes surfaces at
edges, and the shell's residual weight pulls a surface far back (on a synthetic street it gave 15–23 % median error
where the median depth gave 4–6 %).

## Measuring a reconstruction

`evaluation.json` holds three kinds of measure, each with what it can and cannot show.

1. **Held-out frames** (always): every 8th frame rendered at its own pose — PSNR and SSIM over the unmasked pixels,
   and on Falcon runs the range against CARLA's range labels (median absolute relative error, share within 10 %; a
   pixel the reconstruction calls sky or empty where CARLA has a surface within 80 m counts as a 100 % error). These
   poses lie *on* the path, between training frames: an interpolation, the easy case.
2. **Another run beside the path** (`--reference falcon:RUN`): the frames of another Falcon run of the same town whose
   cameras lie within 1 m along and 15° in heading of the path: **genuine CARLA renders at poses the reconstruction
   never saw**, each with its offset from the nearest training camera (lateral to the right, along, vertical), grouped
   by lateral offset. The other run's moving actors are masked.
3. **Lateral shifts against depth-warped CARLA frames** (`--warp-shift M`, both sides, and 0 as a control): no CARLA
   server is started, so the reference at a shifted pose is the held-out frame's own image forward-warped there
   through CARLA's own range labels — exact geometry and each surface's true colour. The warp splats the
   full-resolution frame (four samples a 640 × 360 pixel), keeps the nearest surface per pixel and averages the
   samples on it; it leaves out pixels no sample reaches (newly visible areas, the strip outside the source's view),
   pixels that show a far surface through a gap of a nearer one (a surface 10 % nearer within 2 px on *both* sides —
   a slanted road or wall is nearer on one side only), and pixels from moving actors. It cannot model view-dependent
   shading or anything in the left-out areas, so **its own error is measured** (`--warp-check DIST_M` warps frames to
   the real frame DIST_M further along the path; `--warp-check falcon:RUN` to another run's real frames beside the
   path).

PSNR and SSIM of masked comparisons: left-out pixels take the render's value in the reference (no error) and are not
averaged. SSIM is Wang et al. 2004's (11 × 11 Gaussian window, σ 1.5, valid region), equal to scikit-image's
`structural_similarity` (tested). **LPIPS is not computed**: it needs a learned network and its calibration weights,
and none is pinned in Falcon.

## Measured

On 2026-10-29, RTX 5080, defaults (7000 steps, 640 px wide, seed 0), each run holding Falcon's GPU lock. PSNR in dB;
"depth" is the median absolute relative error of the range against CARLA's range labels and, in brackets, the share
of pixels within 10 %. Two Falcon drives recorded on a real CARLA 0.9.16 server (Town10HD_Opt, ClearNoon, 1280 × 720)
and a KITTI raw drive.

**A Falcon drive with lidar** (`m5/drive`: 160 frames at 20 Hz, 57 m with a right turn, 32-channel lidar; started
from the lidar's static returns): 783 s of training (8.9 steps/s), peak GPU memory 2.1 GiB, 440 444 Gaussians (the
600 000 cap was reached at step 3000 and pruning after the opacity reset brought it down).

| measure | frames | PSNR | SSIM | depth | |
|---|---|---|---|---|---|
| held-out frames (on the path) | 20 | 28.97 | 0.895 | 7.3 % (57 %) | |
| another run on the same path (a smoke drive, mean \|lateral\| 0.03 m; masked by class: it has no instance labels) | 40 | 28.88 | 0.890 | — (no range labels) | genuine renders |
| shifted 0 m (the warp's control) | 20 | 28.80 | 0.894 | 6.7 % | 94 % of pixels defined |
| shifted ±0.5 m | 2 × 20 | 27.01 | 0.875 | 6.6 % | 91 % |
| shifted ±1 m | 2 × 20 | 25.42 | 0.854 | 6.4 % | 87 % |
| shifted ±2 m | 2 × 20 | 23.32 | 0.818 | 5.8 % | 81 % |
| *the warp itself*: 0.5 m / 1 m / 2 m forward against the real CARLA frame there | 20 each | 32.0 / 29.8 / 27.2 | 0.963 / 0.949 / 0.945 | — | 96 / 95 / 86 % |

(Shifted rows: the average of the right and left shifts, against depth-warped CARLA frames.) A 1000-step run of the
same drive scored 23.3 dB held out but 3.3 % depth: density control buys appearance and costs some geometry.

**A Falcon drive without lidar** (`m5/lead`: 240 frames, 81 m straight behind a lead car, which is masked; started
from structure from the known poses: 40 frame pairs, 31 434 matches, 14 763 triangulated, 12 845 points): 643 s
(10.9 steps/s), peak 2.0 GiB, 266 520 Gaussians. Another recording, `gt/town`, drove the same street **3.50 m to the
side** (the next lane, same weather): genuine CARLA renders far off the path.

| measure | frames | PSNR | SSIM | depth |
|---|---|---|---|---|
| held-out frames | 30 | 26.80 | 0.858 | 9.9 % (50 %) |
| another run on the same path (`real/run/anchored`, 640 × 360, mean \|lateral\| 0.03 m) | 40 | 25.02 | 0.804 | 8.3 % (52 %) |
| shifted ±0.5 m (warped references) | 2 × 30 | 23.52 | 0.789 | 7.6 % |
| shifted ±1 m (warped references) | 2 × 30 | 21.93 | 0.743 | 7.3 % |
| **the next lane, 3.50 m to the right (genuine CARLA renders)** | 40 | **16.81** | **0.584** | 18.7 % (44 %) |
| *the warp itself*: 1 m forward / 3.5 m to the side, against the real frames | 30 / 30 | 28.8 / 19.2 | 0.947 / 0.849 | 92 / 59 % of pixels |

**KITTI raw 2011_09_26 drive 0005** (154 frames of image_02, 1242 × 375 worked at 640 × 193, 69 m; started from
the velodyne): 631 s (11.1 steps/s), peak 1.9 GiB, 359 911 Gaussians; held out (20 frames): **20.40 dB, SSIM 0.687**.
The training images themselves are fitted only to about 18–25 dB. Not masked (no tracklets read): the cyclist and
the moving cars in this drive are learned from as if static and leave ghosts; real textures, the camera's exposure
changes and the OXTS poses' own error (taken as given: no pose refinement) are other likely causes — not separated
here. No depth reference exists for KITTI in Falcon (the velodyne is the starting point, not a held-out truth).

What the numbers say:

- **On the path** a reconstruction reproduces CARLA's images to about 27–29 dB (another run of the same path: the same
  level), with the range within 10 % on half to three fifths of the pixels.
- **Off the path it degrades with every metre** (against the 0 m control): −1.8 and −3.4 dB at 0.5 m, −3.4 and −4.9 dB
  at 1 m (the lidar and the structure-from-poses drive), −5.5 dB at 2 m (lidar), and at 3.5 m down to 17 dB, where
  whole regions are wrong — road the lead car hid in every training frame was never observed and renders as a dark
  blur, and surfaces seen only at grazing angles smear. A lidar start held up
  better off the path than structure from poses (−3.4 dB at 1 m against −4.9 dB) — on different drives, so this is
  not a controlled comparison.
- **The warped references are trustworthy only at small shifts**: the warp reproduces a real frame 0.5–1 m away to
  29.8–32.0 dB, well above the reconstruction's 22–27 dB there, so up to 1 m the reconstruction's own error
  dominates; at 2 m the two are closer (27.2 against 23.3 dB), and 3.5 m to the side the warp itself reaches only
  19.2 dB on 59 % of the pixels (view-dependent shading, resampling, disocclusions), so it is no reference that far —
  the genuine renders of the next lane are.
- **Interpolation numbers flatter**: a held-out frame lies between training frames about 0.35 m apart on the path
  (20 Hz at about 7 m/s), looking the same way; it is not a novel view in the sense that matters for a shifted ego.

**Voted semantic classes** (the labels a re-rendered run carries), against CARLA's semantic labels over the unmasked
pixels, classes over 0.1 % of them: `m5/drive` held out 96.7 % of pixels right, mIoU 0.84 over 10 classes (poles 0.54,
road lines 0.66, road 0.98); `m5/lead` held out 92.8 %, mIoU 0.74 (12 classes), and in the next lane 3.5 m to the side
84.8 %, mIoU 0.61 (10 classes).

**Re-rendering** `m5/drive` with the ego 1 m to the right, every second frame: 80 frames in 2.8 s (28 frames/s),
21 MB (RGB, range, semantic).

**Determinism**: two 1500-step CUDA trainings of `m5/drive` (density control included, 341 080 Gaussians) gave
identical parameters bit for bit, as did two 600-step ones. On the CPU, two runs with one thread are identical
(tested) and two with two threads were not (largest parameter difference 0.03 after 40 steps).

## Re-rendering a drive from a shifted path

`falcon recon rerender` (the page's *Re-render the log from the shifted path*) writes a falcon-raw run: every frame
(or every n-th, or only the held-out ones) rendered with the ego moved — and/or another mount, field of view and size
— so the harness and every exporter can read it:

| file | what |
|---|---|
| `rgb/<camera>/<step>.jpg` | the rendered images |
| `range/<camera>/<step>.png` | uint16 range × 256 along each ray, 0 = nothing — **the reconstruction's depth, not ground truth** |
| `semantic/<camera>/<step>.png` | Falcon sources with semantic labels only: the Gaussians' voted classes — **an estimate** |
| `frames.jsonl`, `dataset.json`, `drive.json`, `run.json` | the shifted ego (CARLA's convention of the source's world), the camera, what is and is not available, provenance; `lanes.json` copied from a Falcon source |

**Not available**: instance, objects (2D/3D boxes), optical flow, lidar, radar, actors.jsonl. So a **detector cannot be
scored** on a re-rendered run (the harness needs objects) — it can be run on it for its predictions — while a
**segmenter can be** scored against the voted classes (tested end to end through `core.harness`), with the caveat that
those labels are the reconstruction's own estimate.

## Moving objects

Real logs have moving vehicles and people. A reconstruction is of the **static scene**: masked pixels are never
learned from, so moving actors are not drawn — where a mask misses one (KITTI raw, unmasked; a box that does not cover
a whole pedestrian), it leaves a blurred ghost along its path. Parked cars are part of the static scene.

*Compositing moving actors back* (described, not implemented): a replay of the same log (log replay, `--driver log`
with the shifted ego) renders CARLA's actors at their logged poses; CARLA's instance and depth images of that replay
give each actor's pixels and depth, which can be laid over the reconstruction's image wherever the actor is nearer
than the reconstruction's range — which also gives the re-rendered run real objects labels (boxes from CARLA's
instance image). What it would not give: matching appearance (CARLA's meshes against the log's real pixels), lighting
and shadows consistent between the two.

## The Replay page

The *Reconstruction* section of *Replay* works on the imported log: *Train from this log* queues `falcon recon train`
on the log's own source (a Falcon run, KITTI raw or nuScenes; a ROS 2 bag or Waymo log says why it cannot) into
`workspace/recon/<log>/` as a GPU job; once it is done the reconstruction opens, the view renders any frame with the
ego moved −3 to +3 m by a slider (rendered on a worker thread), and the held-out and novel-view numbers and the cost
are shown. *Measure novel views* runs `falcon recon eval` with ±0.5 m and ±1 m warped references and a 1 m warp
check; *Re-render the log from the shifted path* runs `falcon recon rerender` at the slider's offset into the
workspace's runs.

## Limits

- **Extrapolation degrades fast**: the measured numbers above fall with every metre off the recorded path, and areas
  the drive never saw (behind objects, beyond the lidar's range, the far side of a shifted view) are blurred, missing
  or filled by the far shell. A single forward camera sees each surface over a narrow range of directions.
- **The static scene only**; KITTI raw's moving objects are not masked (no tracklets read). What a masked actor hid
  in every frame (the road under a car followed the whole drive) is never observed and renders as a blur.
- **Pinhole cameras only** (fisheye projection is not implemented); one camera per reconstruction; no rolling shutter,
  exposure or white-balance modelling (a real log's auto-exposure changes are learned as view-dependent colour).
- **Depth sorting by the centre's depth** (the paper's): Gaussians can pop when their order changes.
- **Speed**: plain PyTorch is slower than CUDA kernels (numbers above); 7000 steps by default, not the paper's 30 000.
- **Determinism**: not with several CPU threads; on CUDA measured on one machine, driver and torch version only.
- **LPIPS** is not reported; nuScenes and Waymo are not verified on real files (none on this machine); Waymo and ROS 2
  bags are not reconstructed.
- Re-rendered runs have no objects labels; their range and semantic labels are estimates.

## Licences

The rasteriser, the trainer and the density control are Falcon's (GPL-3.0-or-later), written from the papers cited.
Inputs keep their owners' terms: KITTI (CC BY-NC-SA 3.0) and nuScenes (CC BY-NC-SA 4.0) are non-commercial, and so is
anything reconstructed from them; a reconstruction of a Falcon run is Falcon's output.
