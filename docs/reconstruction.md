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
falcon recon from-dashcam --clip ~/Videos/dashcam-videos/Movie_F/clip.MP4 --start 0 --seconds 20
falcon recon eval ~/demo.falcon/recon/drive --warp-shift 0.5 --warp-shift 1.0 --warp-check 1.0 \
    --reference falcon:~/demo.falcon/runs/other-lane
falcon recon render ~/demo.falcon/recon/drive --frame 40 --lateral 1.0 --out view.png --depth-out range.png
falcon recon rerender ~/demo.falcon/recon/drive --lateral 1.0 --out ~/demo.falcon/runs/drive-shifted-1m
falcon recon hybrid ~/demo.falcon/recon/drive --drive ~/demo.falcon/runs/redrive --out ~/demo.falcon/runs/redrive-hybrid
falcon recon hybrid-drive ~/demo.falcon/recon/drive --project ~/demo.falcon.json --lateral 1.0 --seed 7 \
    --out ~/demo.falcon/runs/hybrid-1m
falcon recon info ~/demo.falcon/recon/drive
```

`train` also takes `--iterations` (7000), `--width` (640: images are worked on at most this wide), `--init
auto|lidar|points|sfm|range|random`, `--holdout` (8), `--seed`, `--sh-degree` (3), `--max-gaussians` (600 000),
`--every`, `--camera`, `--device`, `--raster-backend auto|torch|triton` ([the rasteriser](#the-rasteriser-triton-kernels-and-a-pytorch-reference)),
`--no-masks`, `--no-semantic`, `--config FILE`. `render` and `rerender` take
`--lateral` (metres to the right), `--forward`, `--up`, `--yaw` (degrees, CARLA's sense: positive turns right), `--mount
x,y,z,roll,pitch,yaw` (the camera's mount in CARLA's vehicle frame, as in a rig), `--fov` and `--size WxH`. `hybrid`
and `hybrid-drive` ([hybrid worlds](#hybrid-worlds-carlas-actors-in-the-reconstruction)) take `--method
auto|mask|depth`, `--tolerance` (0.05), `--camera`, `--align FILE`, `--render-width` (0: the working width; −1: the
camera's), `--no-measure`, `--device`; `hybrid-drive` also `--world`, `--rig`, `--lateral`, `--seed`, `--vehicles`,
`--walkers`. Exit codes are the CLI's ([cli](cli.md#exit-codes)).

## The method

3D Gaussian Splatting (Kerbl, Kopanas, Leimkühler and Drettakis, *3D Gaussian Splatting for Real-Time Radiance
Field Rendering*, SIGGRAPH 2023): the scene is a set of anisotropic 3D Gaussians — a mean, a scale and a rotation
(Σ = R S Sᵀ Rᵀ), an opacity and a colour as spherical harmonics up to degree 3 — projected to the image through the
local affine approximation of the perspective map (EWA splatting: Zwicker, Pfister, van Baar and Gross, *EWA
Splatting*, IEEE TVCG 2002), sorted by depth and alpha-composited front to back; their parameters are optimised
against the images by gradient descent through that differentiable rasterisation, while *adaptive density control*
clones, splits and prunes them. Falcon's implementation is written from the papers. **The Inria reference code is not
used** (its licence is non-commercial).

## The rasteriser: Triton kernels and a PyTorch reference

This machine has torch 2.9.1 + CUDA 12.8 and Python 3.14 but no CUDA toolkit (no `nvcc`), so gsplat's CUDA kernels
cannot be compiled here, and its prebuilt wheels do not fit: on 2026-10-29 its wheel index offered gsplat 1.5.3 for
CPython 3.10 only, torch 2.0–2.4 and CUDA 11.8–12.4 (PyPI has only the source-compiling wheel). No Python 3.10 is
installed, and torch ≤ 2.4 with CUDA ≤ 12.4 does not run on an RTX 5080 (Blackwell needs CUDA 12.8), so a
project-local venv would not have helped without system changes. Falcon therefore has its own differentiable
rasteriser (`core/recon/raster.py`), written from the papers, with two backends for its hot part — the per-tile
compositing and its gradient:

- **`triton`** (`core/recon/raster_triton.py`): kernels written in Triton (3.5.1 here; MIT licence), which compiles
  them for the GPU itself — no CUDA toolkit needed. The first render with a given channel count compiles for about
  4 s (measured), then they are cached (`~/.triton`). CUDA and float32 only.
- **`torch`**: the same compositing in plain PyTorch tensor operations — the reference, which runs wherever torch runs
  (any CUDA GPU torch supports, or the CPU) in float32 or float64, and against which every gradient is checked by
  finite differences.

`auto` (the default) uses Triton when the tensors are float32 on a CUDA GPU and Triton imports, PyTorch otherwise;
`falcon recon train --raster-backend torch|triton` (setting `raster_backend`) forces one — `triton` fails, saying why,
where it cannot run. Every render of a training (its steps, held-out checks, the semantic vote, the evaluation) uses
the chosen backend, and run.json records it: `raster: {requested, backend, triton (its version), early_stop_transmittance}`.
Viewing, evaluating and re-rendering a reconstruction use `auto`.

How it works:

- **Projection** of every Gaussian (vectorised, torch, both backends): the camera-frame mean, the 2 × 2 screen
  covariance Σ' = J W Σ Wᵀ Jᵀ + 0.3 I (the paper's dilation; x/z and y/z clamped to 15 % beyond the field for the
  Jacobian, as the paper does) and its inverse. The 3 × 3 products are written as broadcast sums, not batched
  matrix multiplications (cuBLAS's batched GEMM on hundreds of thousands of 3 × 3 matrices took ~27 ms of a step).
- **Exact bounds**: a pixel gets α = min(0.99, o·exp(−½ dᵀ Σ'⁻¹ d)) and α < 1/255 is skipped (as in the paper), so a
  Gaussian touches exactly the pixels inside dᵀ Σ'⁻¹ d ≤ 2 ln(255 o) — its axis-aligned box is used to bin it,
  tighter than the paper's 3σ circle and losing nothing.
- **Tiles**: the depth-sorted Gaussians are listed per 8 × 8 pixel tile (a stable sort by tile keeps the depth order).
- **Compositing**: Tᵢ = Πⱼ<ᵢ(1 − αⱼ), wᵢ = αᵢTᵢ, colour Σ wᵢcᵢ, expected depth Σ wᵢzᵢ, alpha 1 − T_N, and on request
  the **median depth** — the z of the Gaussian at which T falls below ½, with that Gaussian's own flags (is it the
  sky?).
  - *Triton*: one GPU program per tile walks the tile's list 32 Gaussians at a time in registers. It **stops a tile
    once every pixel's T < 10⁻⁴** (checked after each 32), as the CUDA implementations stop a pixel: the image differs
    from compositing everything by less than 10⁻⁴ × the largest value left out (10⁻⁴ of a colour; for the expected
    depth, 10⁻⁴ of the farthest depth left out); the median is unaffected.
  - *PyTorch*: tiles with similar list lengths are composited together, each list padded to the chunk's longest, as
    one tensor of (tiles × 64 pixels × Gaussians) — capped at 2²⁴ elements on a GPU — so memory is bounded whatever
    the scene; every chunk runs under activation checkpointing. No early stop: every listed Gaussian is composited.
- **Gradients**: the PyTorch path's are torch's autograd; `tests/test_recon.py` compares every one (means, scales,
  rotations, opacities, colours, spherical harmonics) with central finite differences in float64. The Triton path has
  hand-written backward kernels (the formulas are in `raster_triton.py`), front to back with the suffix sums taken
  from the forward totals, going exactly as far as the forward pass did — so with the early stop its gradient is that
  of the image it rendered. Per-Gaussian sums are taken in a fixed order (no float atomics), so it is repeatable bit
  for bit. `tests/test_recon_triton.py` checks that with the stop off it agrees with the reference — every output and
  the gradients of the 2D means, conics, opacities, features, depths and median features, and end to end of means,
  scales, rotations, opacities and spherical harmonics — to float32 rounding (10⁻⁴ of each gradient's largest value
  allowed; 1–4 × 10⁻⁷ measured), on random scenes and on empty tiles, Gaussians partly off-screen or at the near
  plane, opacities under 1/255, a median never reached, nothing visible, partial tiles and 4/8/16-pixel tiles; and
  that with it on, the Gaussians it never reached get exactly zero gradient.

**Where the time went** (one training-like step — render, L1 + SSIM loss, backward — of a 477 000-Gaussian Town10
reconstruction at 640 × 360 with degree-3 SH; profiled 2026-10-30 with `torch.profiler` and synchronised timers,
another agent's CARLA server on the GPU for part of it, so the shares are indicative): before, compositing was
nearly all of it — ~33 ms forward and most of a 150 ms backward (recomputation under checkpointing and autograd
through the (tiles × pixels × Gaussians) tensors), against 5 ms projection, 2.4 ms SH and 9 ms sort and binning.
With the Triton kernels the compositing took 1.5 ms forward and 4.2 ms backward, and what remained exposed was the
projection's batched 3 × 3 matrix products (~27 ms of the backward) and the SH's sixteen coefficient slices (~14 ms:
each made a full-size zero-filled gradient); written as broadcast sums, projection and SH take ~2 ms each forward
and the whole backward pass fell from 46 to 8.7 ms; a sorted search replaced a `bincount` in the binning (9 → ~4 ms).
A step is then about 13 ms at 266 000 Gaussians and is bound by the host — the launches of the PyTorch operations
around the kernels and the binning's synchronisations — more than by the GPU.

**Speed** (measured 2026-10-30, RTX 5080, torch 2.9.1 + CUDA 12.8, Triton 3.5.1). The three implementations below
were timed **interleaved in one process** (each step of each in turn), so a shared GPU slows them alike; the GPU had
an idle CARLA server of another agent on it (0–17 % utilisation sampled between runs). *original* is the PyTorch
rasteriser as of commit 5a47ea1 (2026-10-29); *torch* is today's reference path (the broadcast-sum projection and SH
above); *Triton* the kernels.

A training step (render, L1 + SSIM loss, backward; no optimiser) of the finished Town10 reconstruction below
(266 056 Gaussians, 640 × 360, degree-3 SH), the median over 88 steps from 44 training views: original **107.0 ms**,
torch **93.1 ms**, Triton **12.95 ms** — 8.3 × and 7.2 × faster.

One 640 × 360 view of random Gaussians (depth uniform over 2–62 m inside 1.2 × the field, scales 0.02–0.3 m,
opacities 0.1–0.9, degree-3 SH; median of 15 after 5 warm-ups), forward and backward, peak GPU memory:

| Gaussians | (Gaussian, tile) pairs | original | torch | Triton |
|---|---|---|---|---|
| 100 000 | 0.87 M | 24.7 + 62.9 ms, 1.11 GiB | 24.2 + 59.7 ms, 1.11 GiB | 6.5 + 4.1 ms, 0.15 GiB |
| 300 000 | 2.65 M | 61.2 + 182.7 ms, 1.34 GiB | 61.4 + 163.4 ms, 1.33 GiB | 7.9 + 5.1 ms, 0.43 GiB |
| 600 000 | 5.23 M | 112.7 + 348.0 ms, 1.66 GiB | 111.0 + 321.0 ms, 1.66 GiB | 14.5 + 6.6 ms, 0.83 GiB |

(These scenes have about twice the pairs per Gaussian of the 2026-10-29 table this replaces, so they are not
comparable with it.) 8-pixel tiles stay the default: 16-pixel tiles roughly halve the pairs but were slower for both
backends (on the 477 000-Gaussian step above the Triton backward took 93 ms with 16-pixel tiles against 8.7 ms, its
256-pixel register tiles spilling). A render for viewing (colour, median range, sky; `Recon.render`) of the same
266 000-Gaussian reconstruction takes **9.1 ms** with Triton and 30.7 ms with PyTorch (the median of 40, 20 poses).
On the CPU (PyTorch only) one view of 300 000 Gaussians took 0.47 s forward and 2.26 s backward (2026-10-29, AMD
Ryzen 9 9900X, 12 threads): CPU training is for tests and small images.

Training throughput, wall time, memory and quality are under [Measured](#measured). The paper reports real-time
rendering (at least 30 frames a second at 1080p) and training in minutes (about 40 for 30 000 steps) on an A6000;
gsplat could not be run here to compare on the same machine, so no same-machine ratio to it is claimed.

## Sources

Read in place; nothing is copied into the project (KITTI's licence, CC BY-NC-SA 3.0, forbids redistributing it
anyway; the tests write their own KITTI- and nuScenes-shaped folders with made-up numbers).

| source | images and poses | moving objects masked | lidar for the start |
|---|---|---|---|
| `falcon:RUN` | a pinhole camera's `rgb/`; the ego pose of `frames.jsonl` and the camera's actual mount (its faults included); K from its field of view | every vehicle and walker (instance labels → objects with a CARLA actor id; without instance labels, the vehicle and person classes — parked map cars too) and the ego's body | every sweep's static returns (actor id 0, not the ego, not weather noise) through `lidar_pose` |
| `kitti_raw:DRIVE` | `image_02/` (or `image_03/`), rectified; OXTS poses exactly as log replay reads them; the date folder's calibration, cam ← world = [I \| K⁻¹·P_rect's offset] · R_rect_00 · T_cam0←velo · T_velo←imu · T_imu←world (the devkit's chain) | **none**: the tracklets are a separate download and are not read, so moving vehicles stay in the images and blur the reconstruction where they pass | `velodyne_points/` (not motion-compensated: KITTI raw's own limit) |
| `posed:DIR` | a posed-image folder (`posed.json`, `images/`, `masks/`, `points.npz`), today written by `from-dashcam` ([From a dashcam](#from-a-dashcam)): each image's pinhole K, world → camera matrix, ego and mount as the folder states them | the folder's per-image masks | none: its `points.npz` (`--init points`) |
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

`--init auto` uses the source's lidar when it has one, else the points a `posed:` folder brings (`points`: a
dashcam's odometry), else **structure from the known poses**: SIFT features (Lowe
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
several CPU threads the backward pass accumulates in a varying order, so runs differ in the last bits; on CUDA
repeated runs were identical with either rasteriser backend (measured below).

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

**Triton against PyTorch, whole trainings** (2026-10-30, RTX 5080, torch 2.9.1 + CUDA 12.8, Triton 3.5.1; the
defaults — 7000 steps, 640 px wide, seed 0 — on the demo-tour project's `20261030-081500-town10-day-lidar`: a
Town10HD_Opt drive recorded on CARLA 0.9.16, 300 frames at 20 Hz over 52 m, 1280 × 720, a 32-channel lidar; 262
frames trained, 38 held out; started from the lidar; `train()` called directly, without Falcon's GPU lock). The GPU
was **shared with other agents' jobs throughout** — CARLA servers and a translator training — so throughput varied
from run to run; each pair was run back to back and their ratio is the number to read:

| pair (UTC) | also on the GPU | Triton | PyTorch | ratio | original (5a47ea1) |
|---|---|---|---|---|---|
| 1 (06:01–06:32) | a CARLA server at 92–99 % utilisation | 55.9 steps/s | 9.08 steps/s | 6.2 × | 7.70 steps/s |
| 2 (06:34–07:16) | a translator training (3.2 GiB), then a CARLA server | 44.7 and 72.0 steps/s (before, after) | 6.18 steps/s | 7.2 × and 11.6 × | 6.87 steps/s |
| 3 (07:19–07:38) | an idle or lightly loaded CARLA server (3.5–4.3 GiB) | 71.9 and 71.3 steps/s (before, after) | 8.13 steps/s | 8.8 × | — |

In pair 3, the least shared: 97 s of training against 861 s, the whole run (reading the images, training, the
semantic vote, the evaluation) 120 s against 892 s; peak GPU memory **1.24 GiB against 2.08 GiB** (the original
2.09 GiB). The fastest Triton run, started while the GPU was briefly at 10 % utilisation, trained at 101 steps/s
(69 s). The Gaussians peaked near 494 000 at step 3000 and ended at about 266 000. Each implementation is repeatable
bit for bit — all six Triton runs wrote the same `.ply`, all three PyTorch runs another and both original runs a
third — so each has one result:

| rasteriser | held-out PSNR (38 frames) | SSIM | depth | Gaussians |
|---|---|---|---|---|
| Triton | 33.65 | 0.933 | 5.6 % (60.0 %) | 266 056 |
| PyTorch | 33.62 | 0.932 | 5.5 % (59.7 %) | 265 477 |
| original (5a47ea1) | 33.59 | 0.932 | 5.1 % (59.4 %) | 266 592 |

The PyTorch path and the original compute the same mathematics and differ only in the order of float32 roundings (in
the projection and the SH), yet they end 0.03 dB and 0.4 points of depth error apart: rounding alone moves a training
this far. Triton — with its early stop — lies within that spread (+0.04 dB, +0.1 point), so the quality is the same
within the noise of the method.

The measurements below (2026-10-29) used the PyTorch rasteriser as it was then (the *original* above). RTX 5080,
defaults (7000 steps, 640 px wide, seed 0), each run holding Falcon's GPU lock. PSNR in dB;
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
identical parameters bit for bit, as did two 600-step ones (2026-10-29, the PyTorch rasteriser). On 2026-10-30 six
7000-step Triton trainings of the Town10 drive above, under different loads from other jobs on the GPU, wrote the
same `.ply` bit for bit, and so did three PyTorch ones (the Triton backward sums each Gaussian's gradient in a fixed
order rather than with float atomics). On the CPU, two runs with one thread are identical
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

Which pixels are "moving" is decided by the source's labels: in a Falcon run, every object with a CARLA actor id —
every vehicle and walker, moving or stopped (a car waiting at a light is an actor too), and the **traffic lights**,
whose state changes — and the ego's body (`data.actor_object_ids`). The map's own meshes stay, **parked cars
included**: in CARLA's towns they are level geometry (`track_id` `level:<k>`, no actor), as static as a building, and
a hybrid world takes them from the reconstruction. Not masked: the actors' **shadows** (no label covers them), which a
moving actor smears faintly into the road; the actor's anti-aliased edge (no dilation).

## Hybrid worlds: CARLA's actors in the reconstruction

A reconstruction holds the static scene; CARLA holds everything that moves. A **hybrid world** puts the two together:
CARLA simulates and renders the actors — traffic reacting to the ego and to itself, scenario actors, the ego's motion,
the traffic lights' states — and the reconstruction gives the rest, so a drive can have the log's own appearance and
CARLA's reactive actors, with CARLA's exact labels (`core/recon/hybrid.py`).

```sh
falcon recon hybrid RECON --drive RUN --out OUT            # a recorded CARLA drive of the same world, composited
falcon recon hybrid-drive RECON --project P.falcon.json --lateral 1.0 --seed 7 --out DIR   # drive it, then composite
```

**Per camera and tick.** CARLA's camera gives the actors' colours; its instance label (objects with a CARLA actor id,
and the ego class — the same rule that masked them out of training) the actor mask; its range label the actors' range.
The reconstruction rendered at the camera's pose gives the background's colour and range (the median depth's). The
pose is frames.jsonl's ego with the camera's actual mount — CARLA's world mirrored, which *is* the reconstruction's
world when it was made from a Falcon run (no alignment to estimate). Two rules:

* **`mask`** — CARLA's pixel wherever the actor mask holds. CARLA's render has already decided, exactly, which surface
  is nearest at every pixel against CARLA's static geometry, and for a reconstruction of a CARLA run that geometry is
  the reconstruction's scene. Actor pixels are then CARLA's own (identical to its frame); only background pixels are
  the reconstruction's.
* **`depth`** — CARLA's pixel where the actor mask holds *and* the actor's range ≤ the reconstruction's × (1 +
  tolerance) (or the reconstruction has nothing there): the per-pixel depth test a reconstruction whose geometry is
  not CARLA's needs, since its static surfaces may hide an actor CARLA's do not (it cannot show an actor CARLA's own
  geometry hides). An actor pixel it hides takes the reconstruction's labels (voted class, instance 0, its range).

`--method auto` (the default) is `mask` for a reconstruction of a Falcon run used as it is, `depth` with `--align`
(a JSON 4 × 4, the reconstruction's world ← CARLA's mirrored world, rigid: any reconstruction whose frame is aligned
to a CARLA world).

**The background is rendered at the reconstruction's working width** and upsampled (bicubic; range and classes
nearest) for a wider camera (`--render-width -1` renders at the camera's own size). A reconstruction fitted at 640 px
rendered directly at 1280 × 720 shows gaps between its Gaussians — their 0.3 px dilation is in pixels, so they are
half as wide relative to the scene: on the Town 10 reconstruction's 38 held-out frames, **27.31 dB at 1280 × 720
rendered there against 30.04 dB rendered at 640 × 360 and upsampled** (28.75 against 33.68 dB compared at 640 × 360).

**The run** is made from the CARLA drive as a [translated run](translation.md) is: every per-frame file hard-linked
(copied across file systems; a hard link is never written through — a changed file is written beside it and renamed
over it), the run's records copied, each composited camera's rgb written new (JPEG, quality 92 as the drive worker
writes). **Every label is CARLA's** — semantic, instance, range, objects, flow, lidar: exact for the world at that
tick — except the pixels the `depth` rule relabels. The reconstruction's voted classes are not written; they are
measured against CARLA's (below), and CARLA's are kept because they are exact and agree with every other label of the
frame, where the votes are an estimate (91–95 % right on the background below). run.json keeps the drive's (its
clock, rig, settings) with worker `recon.hybrid` and a `hybrid` block: the source drive, the reconstruction and its
`point_cloud.ply`'s SHA-256, the method and rule, the alignment, settings, what was measured; dataset.json a `hybrid`
block; `hybrid.jsonl` a row per frame; `hybrid-previews/` three strips (CARLA, hybrid, reconstruction alone, the
pixels taken from CARLA).

**`hybrid-drive`** records the CARLA drive first: a [log replay](log-replay.md) of the reconstruction's source drive
with the ego on its logged path (driver `log`, moved `--lateral` metres to the right; a shifted path may leave the
lanes, which the replay is told to allow), none of the logged actors, and the traffic manager's traffic — the
source's numbers unless `--vehicles`/`--walkers` — seeded by `--seed` (default the source's + 1), in the source's
weather; then it stops Falcon's server and composites on the GPU (`DIR/carla`, the truth, and `DIR/hybrid`). On the
*Replay* page: *Hybrid world* in the reconstruction section — *Composite this drive* (a recorded run of the workspace
with instance labels) and *Drive the path with new traffic* (the slider's offset; a CARLA job, then the composite).

**Closed loop.** The traffic is closed-loop (the traffic manager reacts to the ego and to itself); the ego follows
the logged path kinematically, so the camera poses are controlled and it does not yield — in 3 of the 6 re-drives
below (0 m and ±1 m) it touched a traffic-manager vehicle. Falcon's own drivers never read the camera's pixels, so
compositing a recorded drive afterwards gives exactly the images compositing at each tick would; `Compositor.frame()`
is one camera and tick, for a loop whose policy drives on the hybrid image (not wired into the environment worker).

**Not modelled: shadows and lighting between the two worlds.** CARLA's actors cast no shadow on the reconstruction's
background, and the actors are lit by CARLA's sun, the background by whatever lit the source. Darkening under actors
from CARLA's own render difference was considered and not done: it needs CARLA's render *without* the actors at the
same tick (a second render with every actor hidden), and the ratio of CARLA's frame to the reconstruction would put
the ground truth into the measured background and cannot carry over to a reconstruction whose appearance is not
CARLA's. The measured cost of the missing shadows is in the *near actors* column below.

### Measured against CARLA's own frames

On 2026-10-30, RTX 5080 (shared with other jobs), the Town 10 reconstruction of the demo project
(`recon/town10-drive`: a 300-frame, 52.6 m drive at 1280 × 720 with 30 cars, trained at 640 × 360, 476 992
Gaussians, lidar start; 33.68 dB / SSIM 0.941 held out). Because the source is a CARLA run, CARLA's own frame at the
same tick and pose is the ground truth of every hybrid frame. Each re-drive: 300 frames at 1280 × 720 (the source's
camera), seed 7, 30 traffic-manager cars, the source's weather, recorded at 5.2 steps a second. PSNR in dB / SSIM,
the mean over frames, over the whole frame and over the background pixels (no actor in CARLA's frame), at 640 × 360
(both images area-resampled, as the training images were) and at 1280 × 720; *near actors*: background pixels within
12 px of an actor at 1280 × 720 (where a shadow would fall, and the actor's edge).

| drive | whole, 640 | background, 640 | background, 1280 | near actors | voted classes: right / mIoU | depth error | shift |
|---|---|---|---|---|---|---|---|
| the source drive itself, its held-out frames (its own traffic) | 33.21 / 0.940 | 33.20 / 0.939 | 30.04 / 0.880 | 22.52 | 95.1 % / 0.718 | 4.2 % | 0.02 px |
| **(a) the same path, other traffic** (0 m) | 29.72 / 0.930 | 29.53 / 0.926 | 27.42 / 0.870 | 19.61 | 94.9 % / 0.718 | 3.8 % | 0.04 px |
| (b) the ego 1 m to the right | 24.48 / 0.825 | 24.41 / 0.822 | 23.75 / 0.787 | 18.45 | 93.8 % / 0.685 | 3.7 % | 0.23 px |
| the ego 1 m to the left | 24.74 / 0.842 | 24.46 / 0.832 | 23.74 / 0.800 | 18.84 | 93.8 % / 0.677 | 3.5 % | 0.28 px |
| the ego 2 m to the right | 21.35 / 0.756 | 21.29 / 0.753 | 20.93 / 0.741 | 18.29 | 92.5 % / 0.635 | 4.1 % | 0.18 px |
| the ego 3 m to the right (off the lanes: the kerb side) | 19.38 / 0.697 | 19.35 / 0.695 | 19.07 / 0.698 | 16.75 | 90.4 % / 0.631 | 4.5 % | 0.16 px |
| the ego 3 m to the left (the oncoming lane) | 19.47 / 0.737 | 19.42 / 0.734 | 19.20 / 0.729 | 16.44 | 91.9 % / 0.639 | 4.5 % | 0.13 px |

*Actor pixels* are CARLA's own under `mask` — identical to the ground truth by construction (SSIM 1, no PSNR). The
*voted classes* column is the reconstruction's estimate against CARLA's semantic label on background pixels (the
labels written are CARLA's); *depth error*: the reconstruction's range against CARLA's on background pixels within
80 m (median absolute relative error); *shift*: the residual translation between the reconstruction's render and
CARLA's frame (phase correlation of the background, median).

What the numbers say:

* **The alignment is exact**: on the 0 m re-drive every camera that should coincide with a training camera does
  (offset ≤ 10⁻⁴ m, 0.0°), the residual image shift is 0.04 px and the background's range is within 3.8 % (median).
  The camera's exposure did not change with the traffic (one gain per frame between the source's and the re-drive's
  frames on common background pixels: 1.000–1.012, no PSNR gained by it).
* **Even CARLA does not repeat its own background**: the source's own frames against the 0 m re-drive's, at the
  same poses, on pixels that are background in both, score **32.43 dB at 640 × 360 and 30.70 dB at 1280 × 720** —
  other cars cast other shadows. That is the ceiling for any static background under new traffic. The hybrid's 29.53
  dB lies 2.9 dB below it, and that gap is the road the source's own traffic hid: over background pixels the source
  frame at that step also saw, the reconstruction scores 30.80 dB (mean of frames); over the 1.25 % of a frame where
  a source actor stood, **12.39 dB** (pooled) — what a masked actor hid in every training frame was never observed.
* **Training poses are no easier than held-out ones** under new traffic (0 m: 29.53 dB at both; the source's own
  drive: 33.42 against 33.20 dB): the reconstruction is not over-fitted, and new traffic, not interpolation, sets the
  error.
* **Off the path it degrades with every metre**: −5.1 dB at 1 m (either side), −8.2 dB at 2 m, −10.1 to −10.2 dB at 3
  m (both sides) against the 0 m re-drive, on genuine CARLA renders — the same fall as against the depth-warped references
  above (−3.4 to −4.9 dB at 1 m), steeper because these frames also carry new traffic and every frame is off the path.
  The voted classes hold up better (95 → 90 % right).
* **The missing shadows show**: the background within 12 px of an actor scores 7.8 dB below the rest at 0 m (19.61
  against 27.42 dB at 1280 × 720) and 2.3 dB below it at 3 m, where the reconstruction's own error dominates.
* **A re-drive at 640 × 360** (CARLA rendering at the training width itself) scores 27.13 dB on the background,
  against 27.74 dB for CARLA's own 640 × 360 frames against its 1280 × 720 frames area-resampled at the same poses: a
  renderer's resolution changes its images by as much as the reconstruction's error here, so a re-drive's camera
  should be the source's and the comparison made at the working width.

**`mask` against `depth`** (the depth test as the alternative on the same frames; the share of actor pixels it hides
though CARLA shows them, per frame then averaged — small, distant actors weigh as much as near ones):

| drive | actors, `depth` at 5 % | actor pixels hidden | whole frame, 1280: `mask` / `depth` |
|---|---|---|---|
| the source's held-out frames | 27.00 dB | 7.9 % | 30.10 / 29.83 |
| 0 m | 25.99 dB | 9.4 % | 27.64 / 27.54 |
| +1 m / −1 m | 23.22 / 30.09 dB | 11.1 / 7.9 % | 23.83 / 23.79, 24.02 / 23.98 |
| +2 m | 22.22 dB | 12.4 % | 20.99 / 20.96 |
| +3 m / −3 m | 20.82 / 22.80 dB | 18.5 / 10.5 % | 19.10 / 19.04, 19.25 / 19.23 |

The tolerance (the depth test as a pure measurement; actor PSNR, then the share of actor pixels hidden — per frame
averaged / pooled over all actor pixels):

| tolerance | 0 | 5 % | 10 % | 20 % | 50 % | 100 % |
|---|---|---|---|---|---|---|
| 0 m | 25.48 dB, 19.5 / 1.4 % | 25.99, 9.4 / 0.5 % | 26.94, 8.6 / 0.4 % | 28.40, 6.3 / 0.3 % | 35.37, 1.2 / 0.07 % | 45.07, 0.6 / 0.03 % |
| +3 m | 19.04 dB, 34.8 / 33.3 % | 20.82, 18.5 / 27.7 % | 21.10, 17.3 / 27.3 % | 21.52, 14.7 / 26.4 % | 25.27, 8.8 / 24.5 % | 26.42, 6.8 / 20.6 % |

On the path, what the test hides is little of the actors' area (0.5 % pooled at 5 %) but much of small, distant
actors (9.4 % per frame), where the reconstruction's depth is coarsest. 3 m off the path it hides a quarter of the
actors' pixels even at 100 % tolerance: from a viewpoint the drive never took, the reconstruction puts surfaces
(floaters, smeared geometry seen only at grazing angles) in front of what CARLA shows, at half its range or nearer.

So on a reconstruction of CARLA's own world the depth test only loses: CARLA's occlusion is exact, and the
reconstruction's median depth is a few per cent off everywhere and far off at blurred edges and floaters. `auto`
therefore uses `mask` there; `depth` is for reconstructions whose geometry is not CARLA's, where these numbers say a
5 % tolerance hides a tenth of the actors' pixels, mostly on small, distant cars.

**Cost** (a frame, medians; the GPU was shared with other jobs at about 75 % utilisation, so the 1280 × 720 numbers
spread to 70 ms at the 90th percentile): at 640 × 360, rendering the background 6.0 ms and compositing 0.5 ms; at
1280 × 720 (rendered at 640 × 360 and upsampled) 14.8–20.4 ms and 6.7–6.9 ms. Reading CARLA's frame and labels and
writing the JPEG: 6.7 ms at 640 × 360, 49 ms at 1280 × 720; measuring against CARLA: 12 ms and 245–257 ms. A 300-frame
1280 × 720 re-drive composites and measures in 70–180 s; its CARLA drive took 58 s (5.2 steps a second, labels
included) plus the server's start.

## The Replay page

The *Reconstruction* section of *Replay* works on the imported log: *Train from this log* queues `falcon recon train`
on the log's own source (a Falcon run, KITTI raw or nuScenes; a ROS 2 bag or Waymo log says why it cannot) into
`workspace/recon/<log>/` as a GPU job; once it is done the reconstruction opens, the view renders any frame with the
ego moved −3 to +3 m by a slider (rendered on a worker thread), and the held-out and novel-view numbers and the cost
are shown. *Measure novel views* runs `falcon recon eval` with ±0.5 m and ±1 m warped references and a 1 m warp
check; *Re-render the log from the shifted path* runs `falcon recon rerender` at the slider's offset into the
workspace's runs. *Hybrid world* composites a recorded drive of the workspace, or drives the log's path anew with seeded
traffic at the slider's offset ([hybrid worlds](#hybrid-worlds-carlas-actors-in-the-reconstruction)).

## From a dashcam

```sh
falcon recon from-dashcam --clip ~/Videos/dashcam-videos/Movie_F/clip.MP4 --start 0 \
    --seconds 20 --project ~/FalconWorkspace/Bengaluru.falcon.json --world w-blr
    # → ~/FalconWorkspace/recon/dashcam-<clip>-<start>s
falcon recon from-dashcam --clip … --start 34 --seconds 20 \
    --intrinsics ~/FalconWorkspace/recon/dashcam-…/intrinsics.json
falcon recon place ~/FalconWorkspace/recon/dashcam-… --project P.falcon.json --world W
    # place a posed segment on a world imported later (e.g. by `falcon dashcam route import`)
```

A consumer dashcam gives neither calibrated intrinsics nor poses: only its video and about one GPS fix a second
(metres of error, no orientation). `from-dashcam` (`core/recon/fromdashcam.py`, `core/recon/vo.py`; OpenCV, numpy and
scipy — COLMAP, OpenSfM and pycolmap are not used) poses a short segment (10–30 s of moving, daytime driving) itself and
hands the posed images to the reconstruction above as a `posed:` source. The clip's frames and GPS come from the
dashcam ingest (`core/dashcam.py`, [dashcam](dashcam.md)). Other flags: `--iterations`, `--width`, `--seed`,
`--device`, `--image-every` (3: every third decoded frame is a posed image), `--vo-width` (960), `--no-train`,
`--keep-raw`, `--config FILE` (every setting of `fromdashcam.DEFAULTS`). The page: *From a dashcam* in the
*Reconstruction* section of *Replay* (a clip, a window, a world; the result opens in the section's view).

### The method

1. **Frames and GPS.** Every frame of the window (30 a second) is decoded through `core.dashcam` (NVDEC when it
   works): grey copies at 960 × 540 for the odometry, every third frame kept at 1920 × 1080 for the images. The GPS
   fixes in the window (±2 s) on the video's clock, with their speed over ground.
2. **What not to track.** The bonnet and the camera's burnt-in text: pixels whose mean absolute difference between
   frames 0.5 s apart stays under 3 grey levels, connected to the bottom edge — plus a fixed bottom band of 9 %,
   because this camera's bonnet reflects the moving scene and so changes (heuristic, measured on the user's clips: the
   bonnet's top is at 91 % of the height). The posed images are cropped above it. A corridor of the road ahead
   (34–66 % of the width, the bottom 45 %) is left out of the odometry only (not of the images): the traffic the
   camera follows sits there (heuristic; on the first segment it cut the odometry-only GPS residual from 16.3 m to
   5.6 m RMS).
3. **Intrinsics** (unknown). The overlay names the camera model (a consumer 4K dashcam); retailers state a 170° field for its
   front camera, with no manufacturer's sheet found — a fisheye-style diagonal figure that says nothing about a
   pinhole's focal length, so it is not used. Instead:
   * **k1 by the plumb-line method** (Brown 1971): Canny edge chains that are straight for some k1 and lie off the
     centre; the k1 that straightens them best, its spread by a bootstrap. It is used only when it differs from 0 by
     more than twice that spread (the grid stops where a barrel term would fold the image back). The principal point
     is assumed at the image centre and k2 = 0.
   * **f by a focal search**: the odometry (below) run with horizontal fields of 90°–130° (`hfov_grid`). On a
     segment that turns by 30° or more (GPS course) f is measured directly: a rotation moves the image by f × the
     angle, so the camera's yaw change seen by an odometry run with a wrong f is scaled, and each candidate gives
     f × (camera yaw change / course change); their median is used. Otherwise the candidate whose track fits the GPS
     best (the median residual of the odometry-only alignment, step 6) is taken — a weak criterion, since f is barely
     determined by forward driving: on KITTI raw drive 0015 (a calibrated, global-shutter camera; 363 m) the
     odometry's error against OXTS changed by under 0.3 m with f off by −20 … +25 %, and the reprojection error not at
     all. Self-calibration in the bundle adjustment (f and k1 free; `vo.self_calibrate`) recovers f to ±5 % on
     synthetic streets with rotation (tests) but was unstable on the dashcam footage (f wandered between rounds and k1
     ran off), so it is not used by default.
   * `--intrinsics FILE` reuses a run's `intrinsics.json` (native 3840 × 2160 pixels) for other segments of the
     same camera.
4. **Visual odometry.** KLT tracks (Shi–Tomasi corners, forward–backward check, an essential-matrix RANSAC between
   consecutive frames drops points leaving their epipolar lines), an essential-matrix start, keyframes by parallax,
   PnP for every frame, triangulation at keyframes, a local bundle adjustment over 12 keyframes, a global one at the
   end. The bundle adjustment is Levenberg–Marquardt with analytic derivatives, a Huber loss and the Schur complement
   (`vo.bundle_adjust`): `scipy.optimize.least_squares` was tried first, and with a Huber loss its trust-region
   solver stalled (58 → 40 px RMS in 100 evaluations on a synthetic 8-camera problem whose solution is 0.26 px).
   **The GPS speed holds the scale**: the speed over ground integrated between two frames is the distance travelled
   (σ = 3 % + 5 cm, heuristic: Doppler speed is good to ~0.1 m/s, its timing to ~0.5 s); the map is metric from the
   first keyframe pair, every bundle adjustment holds consecutive keyframes to those distances, and each frame's pose
   is first predicted at that distance, the map points more than 6 px from the prediction left out (and dropped after
   three such frames). Without this the odometry was useless on the user's footage: points on a tanker the car was
   overtaking — moving along their epipolar lines, so no epipolar test sees them — pulled the PnP towards standing
   still and the scale fell about sixfold in five seconds. The GPS *positions* are not used by the odometry.
   **Wipers.** A wiper's sweep wipes out most tracks in one or two frames (on the user's footage in drizzle a sweep
   every ~3.4 s, each 3–5 frames with more than half the tracks lost and a jump in the mean grey difference to 20–28
   levels against a median of 3.3). Such a step (more than `bridge_below` = half the tracks lost at once) holds the
   tracks of the frame before and follows them straight into each next frame, each point started where its image
   velocity before the sweep puts it, until `bridge_accept` = 30 % come through the forward–backward and epipolar
   checks (at most `bridge_frames` = 10 frames); the frames in between keep no tracks and the odometry skips them
   (at most `max_skip` = 10 frames after the last placed one, also when PnP fails), so the map carries on across the
   sweep. All four are heuristic. When the odometry still stops early, the summary says where (`lost`) and the rest
   can be its own segment; when it places too little to align (under 30 frames or 3 s), the error says how much it
   placed, the points it tracked a frame and the speed.
5. **Up and the road.** "Up" is the cameras' mean up axis made perpendicular to the direction of travel; the road
   plane is fitted (RANSAC, 12 cm) among the points 0.5–3 m below the cameras and within 8 m of their path; z = 0 is
   that plane and the camera's height above it a check on the scale (a windscreen camera sits about 1.2–1.5 m up).
6. **The GPS alignment** (`vo.align_to_gps`): a robust similarity (RANSAC, then Huber-reweighted Umeyama) from the
   levelled odometry to the fixes in a local east-north-up frame (transverse Mercator about the first fix). **Its
   residuals are the honest measure**: the odometry never saw the positions. The GPS latency against the video
   (±0.5 s per the ingest) is searched over ±1 s but only believed when it improves the median residual by 25 % *and*
   50 cm: on a straight road at constant speed (or an arc) a latency is indistinguishable from an offset along the
   track, and on a synthetic straight drive the float32 quantisation of the coordinates alone moved the best lag to
   0.6 s.
7. **GPS-aided poses** (`vo.gps_aided_ba`): a final bundle adjustment of the metric, geolocated map with every keyframe
   pulled towards its interpolated fix (σ 3 m east and north) and towards the camera height above z = 0 (σ 2 m: flat
   roads assumed). These are the poses the reconstruction uses; their residuals against the fixes are reported too
   but are **not independent** (the fixes are in the adjustment). The frames between keyframes are re-placed by PnP,
   or interpolated between the keyframes when PnP strays from their path.
8. **Posed images.** The kept frames undistorted to a pinhole and cropped to the valid area above the bonnet (960 px
   wide), each with a mask of moving pixels: dense Farnebäck flow to the frame 0.1 s later more than 2 px off its
   epipolar line, opened and dilated — catches crossing traffic, not traffic moving along the camera's line; a frame
   whose mask covers more than 40 % (the test failed there) is left unmasked. The odometry's points, coloured from the
   images, start the Gaussians (`--init points`, the default for this source).
9. **Reconstruction and novel views**: `core.recon.train` with the settings above (7000 steps, 640 px); held-out every
   8th image as always, also split at the horizon (pixels whose ray points below / above the horizontal — the sky is a
   large, smooth part of a dashcam frame and flatters the whole-image numbers); then a few held-out frames rendered
   with the ego 0.5 m and 1 m to either side and the camera 0.5 m up and down (`renders/novel-*`). Nothing was
   recorded at those poses, so they are judged by eye.
10. **On the map**: the geolocated track converted into the imported world's frame (`core.dashcam.to_world`), the share
    inside the import box, the distance and bearing from its centre and, when the track passes within 50 m of the
    imported lanes, the nearest imported road's name (graph.json), log replay's map match of the track
    (`core.logreplay.mapmatch`: on a lane in its direction, lateral offset) and the share of the reconstruction's
    road-plane points (within 15 cm of z = 0 and 25 m of the path) that fall within an imported driving lane's half
    width + 0.25 m — a heuristic: kerbs, verges and parked cars on the plane count against it, unused lanes for it.

### Measured on the user's dashcam

On 2026-10-29, RTX 5080, defaults (seed 0). The clips (a consumer 4K dashcam, 3840 × 2160 HEVC at 30 frames
a second, GPS at 1 Hz) are private: only numbers are given here. **No clip lies inside the imported 1 km² of central
Bengaluru** (`w-blr`): every fix is several kilometres outside it, and the clips in the city were recorded after
sunset. Three daytime segments of a highway drive well outside the imported area were taken, so no lane alignment can
be measured for them (their placement in `w-blr`'s frame is in each run's `dashcam.json`).

| segment | speed | GPS course change | traffic |
|---|---|---|---|
| A: one clip, 0–20 s | 40–74 km/h | 17° (a gentle curve) | a car and a tanker ahead, overtaking |
| B: the same clip, 34–54 s | 87–101 km/h | 61° (a long curve) | light; a motorcycle |
| C: another clip, 20–40 s | 87–94 km/h | 1° (straight) | a car ahead |

**Intrinsics.** f was measured on B (the turn): the five candidates' turn estimates were 586, 561, 553, 530 and 541 px
at 960 px wide (a separate calibration run on the same segment with candidates of 75–110° gave 556–586 px): **f = 553
px at 960 wide, 2213 px at 3840 (a pinhole field of 81.9° horizontal), ±3 % (the spread of the estimates)**. A and C
reuse it (`--intrinsics`). As an independent check, on A — which did not contribute to the estimate — the camera's yaw
change came out −17.6° against a GPS course change of −17.35° (with the first guess of 100°, f = 403 px, it had been
−24.8°). The plumb-line k1 was +0.04 ± 0.05 on A (not used: not different from 0) and at the edge of its grid on B
(not used): **k1 = 0 assumed** — the frames show no visible barrel distortion (the camera likely corrects it itself);
k2 and the principal point were not estimated.

**Poses against the GPS** (metres; the odometry-only alignment is independent of the fixes, the GPS-aided one is not;
the GPS has metres of error of its own):

| segment | path (GPS) | odometry only: RMS / median / max | scale | GPS-aided: RMS / median / max | camera yaw vs course |
|---|---|---|---|---|---|
| A | 305 m | **3.5 / 2.8 / 6.7** | 1.061 | 4.4 / 3.6 / 11.6 | −17.6° / −17.35° |
| B | 525 m | **8.1 / 4.0 / 27.6** | 1.049 | 7.3 / 3.8 / 26.0 | +64.9° / +61.3° |
| C | 470 m | **11.3 / 3.0 / 28.6** | 0.958 | 4.0 / 3.5 / 8.5 | −10.3° / −0.3° |

The scale (1 = the GPS speed's metres) stayed within 6 % thanks to the speed priors. The medians are at the GPS's own
level; the maxima are not: on B and C the odometry's heading drifts (on C 10° on a straight road), and the summed
frame-to-frame path is 2–26 % longer than the GPS's — the poses jitter. On B, f = 572 px (the 80° candidate of the
calibration run) gave 4.3 m RMS where 553 px gave 8.1 m: the odometry on this footage is that sensitive. The
reprojection error of the GPS-aided poses: median 0.53–0.71 px, 95–97 % of the observations within 4 px (960 px
wide). The road plane was believed (1.94 m camera height, high for a car) on C only; A and B use the assumed 1.3 m.

**Reconstructions** (640 × 323, 7000 steps; 175 training and 25 held-out images 0.1 s apart, i.e. 1.2–2.8 m):

| segment | held out: PSNR / SSIM | below the horizon | above it (mostly sky) | Gaussians | training |
|---|---|---|---|---|---|
| A | 28.27 / 0.885 | **24.98 / 0.693** | 29.95 / 0.934 | 57 933 | 366 s (19.1 steps/s), peak 1.36 GiB |
| B | 26.37 / 0.851 | **22.97 / 0.654** | 28.26 / 0.906 | 45 561 | 429 s (16.3 steps/s), peak 1.37 GiB |
| C | 28.41 / 0.891 | **23.19 / 0.656** | 31.72 / 0.941 | 38 725 | 369 s (19.0 steps/s), peak 1.34 GiB |

Below the horizon — about 23 % of each image here, because this camera looks up (the direction of travel is 17° below
its optical axis) — the reconstructions reach 23–25 dB and SSIM 0.65–0.69, against 27–29 dB / 0.85–0.90 on CARLA
drives above: the road and the verges are recognisable but soft, poles and wires are mostly lost, and the traffic the
masks miss (cars driving ahead) leaves dark, smeared ghosts on the road. The renders 0.5 m and 1 m to the side and
0.5 m up and down look like the same blurred scene from slightly elsewhere; nothing was recorded there, so no number
can be given. Few Gaussians grew (39–58 k, against 270–600 k on CARLA): the odometry's 5–11 k points are a sparse
start and the far shell takes the sky.

**More segments, placed on the route's world** (2026-10-29, the same intrinsics, f = 2212.88 px given). Five 20 s
segments along a 10-minute stretch of the user's footage (dry highway into rain), placed with `falcon recon place` on
the world `falcon dashcam route import` built for that stretch ([dashcam](dashcam.md#driving-your-route-in-simulation)),
so the lane check that could not run before now runs:

| segment | posed frames | odometry only: RMS / median / max (m) | GPS-aided | held out: whole | below the horizon | above | cameras on a lane in their direction, lateral median |
|---|---|---|---|---|---|---|---|
| dry highway, trucks | 597 / 600 | 7.1 / 2.2 / 15.7 | 5.1 / 2.6 / 16.0 | 28.31 / 0.876 | **24.10 / 0.699** | 30.34 / 0.916 | 91.5 %, 0.99 m |
| dry highway, cars | 298 / 600 (lost) | 1.6 / 1.0 / 2.5 | 7.1 / 5.7 / 13.7 | 28.60 / 0.893 | **24.48 / 0.712** | 31.21 / 0.937 | 100 %, 0.62 m |
| dry into drizzle | 589 / 600 | 4.8 / 2.6 / 11.2 | 9.7 / 4.5 / 23.6 | 27.75 / 0.890 | **22.86 / 0.665** | 31.30 / 0.944 | 55.9 %, 1.26 m |
| drizzle, wipers¹ | 439 / 600 (lost) | 3.3 / 2.2 / 6.6 | 4.0 / 3.1 / 9.6 | 30.33 / 0.911 | **26.16 / 0.749** | 32.58 / 0.948 | 2.3 %, 3.00 m |
| rain, wipers | 494 / 600 (lost) | 2.6 / 2.3 / 4.3 | 3.2 / 2.6 / 5.4 | 29.88 / 0.912 | **26.63 / 0.749** | 31.55 / 0.949 | 100 %, 0.90 m |

¹ Posed by the first version of the bridge (tracks held at 50 %, no velocity prediction); the settings described
in step 4 place 146 frames of it. "Lost" on the dry segment with cars: no frame there was disrupted (no sweep); the cause was not
established (a car passing close is likely).

- **The lane check**: the reconstructions' camera paths lie on a lane in their direction for 56–100 % of their
  positions, a median 0.6–1.3 m from its centre, on four of five segments; one (drizzle) sits a median 3.0 m off,
  in the neighbouring lane — the GPS-aided poses carry the GPS's offsets (the raw GPS on this stretch: 59.9 % on a
  lane in direction, median 1.39 m). The first segment of the earlier table, placed the same way on a world of its
  own clip: 76.7 %, 1.05 m. The road-plane points tell little: 32–318 points within 0.15 m of the plane a segment,
  6–53 % of them on a lane (heuristic; the plane was believed on one segment of five, the others use the assumed
  1.3 m camera height).
- **Rain**: the wet segments score higher held out (26 dB below the horizon) because a wet, grey scene has less
  detail to get wrong, not because they are better reconstructions. The wipers are bridged (step 4): 68–106 swept
  frames skipped a segment. The odometry still lost its track on three segments (the rest of a segment can be its
  own), and **of five more wet segments further on — heavy rain, a town, slow and stopped traffic — none could be
  posed** (8–50 frames placed before the track was lost: too little to align). Before the bridge, every segment
  with a wiper failed within 0.6 s. The bridge's settings are fragile: with `bridge_accept` 0.4 or 0.5 instead of
  0.3 one wet segment lost its track at once and another went from 146 to 105 or 480 frames.
- **Costs**: 6.8–12.0 minutes a segment (training 380–648 s, some of it waiting for the GPU lock while CARLA ran), peak
  memory 2.7 GB.

**Costs** (per 20 s segment, 600 frames): decoding 3.9–4.2 s (NVDEC, 1920 × 1080); tracking 4.0–4.8 s; the
odometry 8.8–10.1 ms a frame beside its local bundle adjustments, which took 0.12–0.14 s each (12 keyframes, one per
keyframe, 120–172 of them); the global adjustment 13–24 s (two rounds, 30–43 k observations); the GPS-aided one
6–17 s; the focal search 187 s (five candidates, B only); undistorting and writing the images 5–6 s; the
reconstruction 6–7 minutes. The whole run took 7.0–7.5 minutes (A, C) and 11.2 minutes (B, with the search), 2.7 GB of
memory at most.

Output (`--out`): `dashcam.json` (every number of every step), `intrinsics.json`, `posed/` (`posed.json`, `images/`,
`masks/`, `points.npz` — the `posed:` source), and the reconstruction's own files (`point_cloud.ply`, `views.json`,
`evaluation.json`, `renders/`, `run.json`). **The footage is private**: nothing is copied anywhere but `--out`, and the
tests use synthetic clips (a textured street rendered along a known track, encoded with ffmpeg, given a GPS index in
the camera's layout).

## Limits

- **Extrapolation degrades fast**: the measured numbers above fall with every metre off the recorded path, and areas
  the drive never saw (behind objects, beyond the lidar's range, the far side of a shifted view) are blurred, missing
  or filled by the far shell. A single forward camera sees each surface over a narrow range of directions.
- **The static scene only**; KITTI raw's moving objects are not masked (no tracklets read). What a masked actor hid
  in every frame (the road under a car followed the whole drive) is never observed and renders as a blur.
- **Pinhole cameras only** (fisheye projection is not implemented); one camera per reconstruction; no rolling shutter,
  exposure or white-balance modelling (a real log's auto-exposure changes are learned as view-dependent colour).
- **Depth sorting by the centre's depth** (the paper's): Gaussians can pop when their order changes.
- **Speed**: the Triton backend made a training 6–12 times faster than plain PyTorch (8.8 × in the least shared
  pair, measured above); a step is now bound by the PyTorch operations around the kernels (projection, SH, sorting,
  binning, the optimiser) more than by the compositing. 7000 steps by default, not the paper's 30 000. The Triton
  backend is CUDA and float32 only, stops a tile at T < 10⁻⁴ (a difference below 10⁻⁴ of a colour), and compiles its
  kernels (~4 s) at first use.
- **Determinism**: not with several CPU threads; on CUDA measured on one machine, driver and torch version only.
- **LPIPS** is not reported; nuScenes and Waymo are not verified on real files (none on this machine); Waymo and ROS 2
  bags are not reconstructed.
- Re-rendered runs have no objects labels; their range and semantic labels are estimates (a hybrid world's are
  CARLA's).
- **Hybrid worlds**: no shadows or lighting between CARLA's actors and the reconstruction (measured: the background
  near actors is 7.8 dB worse on the path); the actors look like CARLA's meshes, not the log's; road a source actor
  hid in every frame renders as a blur (12.4 dB there); the depth test, needed for a reconstruction whose geometry is
  not CARLA's, hides a tenth of small actors' pixels on the path and a quarter of all actor pixels 3 m off it; the
  re-drive's ego follows the logged path kinematically and does not yield; compositing runs after the drive (a policy
  that drives on the hybrid image is not wired in); pinhole cameras only (fisheye cameras of the rig keep CARLA's
  image). Measured on one CARLA reconstruction (Town 10); a reconstruction of real footage aligned to a CARLA world is
  accepted (`--align`) but not measured, and CARLA's frame is no ground truth for it.
- **From a dashcam**: monocular odometry with the GPS speed for scale; the poses are only as good as the measured
  residuals (metres against the GPS, which has metres of error itself). Not modelled: rolling shutter, auto-exposure,
  motion blur, rain drops on the windscreen; wipers are bridged (step 4) but rain defeats the odometry more often than
  not (measured below), and a frame with part of a blade in view still trains the reconstruction. Traffic moving along the camera's line is neither masked nor
  fully kept out of the odometry (the corridor and the speed gate are heuristics), and it leaves ghosts in the
  reconstruction. The principal point is assumed central and k2 = 0; f is measured only on a turn. Flat roads are
  assumed for "up" and the height prior; the GPS latency is taken as 0 unless the fit clearly says otherwise. A
  segment far from an imported world cannot be checked against its lanes.

## Licences

The rasteriser, the trainer and the density control are Falcon's (GPL-3.0-or-later), written from the papers cited.
Inputs keep their owners' terms: KITTI (CC BY-NC-SA 3.0) and nuScenes (CC BY-NC-SA 4.0) are non-commercial, and so is
anything reconstructed from them; a reconstruction of a Falcon run is Falcon's output.
