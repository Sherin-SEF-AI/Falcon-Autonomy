# The bird's-eye surround view

A rig's cameras — fisheyes or pinholes, two or more — stitched into one view of the ground around the car from above,
as a surround-view (SVM) ECU shows it. The tables are built from the rig's own poses and lens models, so the same
stitch runs on a recorded Falcon run, live in the GUI, and — exported — on the vehicle with the Apache-2.0 reference
stitcher. The code is `core/bevstitch.py` (ported from Vantage's `core/stitch.py`); the page is *Rigs ▸ Surround view*.

```sh
falcon bev stitch ~/FalconWorkspace/runs/svm-drive --balance --seams      # RUN/bev/<step>.png and bev.json
falcon bev export surround --project ~/demo.falcon.json --out ~/bev-tables # tables + stitcher/ for the vehicle
falcon bev export rig.json --out ~/bev-tables --res 0.01                    # a rig JSON file (falcon.rig/1)
falcon bev bench --size 1920x1080 --frames 100                              # frames per second
```

`stitch` takes `--every`, `--step N` (repeatable), `--balance` (exposure gains per frame), `--seams` (drawn on the
images), `--device auto|cuda|cpu|torch-cpu`; every command takes `--res` (ground metres per pixel, 0.02), `--margin`
(metres of ground beyond the body on every side, 4) and `--blend` (the seam ramp's width, m: the rig's own when it came
from Vantage, else 0.2). `export --no-runtime` writes the tables only. Exit codes are the CLI's.

## How it is built

**The ground grid** is in CARLA's vehicle frame (x forward, y right, z up, origin on the ground under the bounding
box's centre), the ground the plane z = 0; the image shows the front at the top and the left on the left (row 0 is the
front edge, column 0 the left edge). For each ground pixel and each camera, the ray from the camera to the point is
taken into the camera's optical frame by its pose and through its lens model (`core.lens.Intrinsics`: rectilinear,
equidistant, equisolid, stereographic, orthographic, Kannala–Brandt): the table holds where the camera samples it, in
pixel-edge coordinates of the sensor's size, or −1. A **pinhole** camera's ground mapping is a homography, H = K [r₁ r₂
t], computed in closed form (`ground_homography`; tested equal to the general path to 10⁻⁶ px).

**What a camera cannot see**: outside its field and image, behind it, and through the **body** — the CARLA car's box
(`core.vantagerig.CARLA_VEHICLES`, its width less the mirrors, as Vantage takes it; a rig imported from Vantage brings
the project's own body) blocks every camera's rays from 0.2 m to the roof, except within 0.3 m of the camera, so a
camera on the body's surface does not block itself. The ground under the body is drawn dark.

**Seams** (azimuths are Vantage's: about the body centre, counter-clockwise from ahead seen from above): a rig imported
from Vantage (`falcon rigs import-vantage`) keeps its variant's seams — the ones set in Vantage, or estimated there —
and its body centre. Any other rig's seams are estimated here by **Vantage's own rule** on its cameras: each camera's
density-weighted mean azimuth over the ground it resolves best within 2 m of the body, then for each neighbouring pair
the ray that maximises the shared, well-resolved ground (`estimate_seams`). A seam hands the sector after it to its
camera b — the same owner `core.rigcompare.owner` gives (*Rigs ▸ compare* scores a detector by it).

**Weights**: the owner by sector, a linear ramp across each seam line (blend_m wide), only where the camera sees the
point; a point its owner cannot see goes to the camera that resolves it best (Vantage's pixel density, px/cm). The
weights sum to 1 wherever any camera sees the ground. A camera without a frame is left out and its ground shared among
the others (the runtime's rule).

**Exposure** (`--balance`, *Balance exposure*): Brown and Lowe's gain compensation over the ground two cameras share
(well-exposed pixels only), per camera and channel, clipped to 0.4–2.5 (a clip is reported). Tested: one camera made
35 % brighter is brought back to within 5 % of the others, and the seam mismatch falls fourfold.

## The stitcher

| backend | what | when |
|---|---|---|
| torch on CUDA (half precision) | each camera's table is a `grid_sample` grid (pixel-edge u → 2u/W − 1, so frames of any resolution with the sensor's aspect are read right); every camera of one image size in one batched `grid_sample`; weights and sums on the GPU (sums in float32) | `--device auto` with a CUDA GPU, or `cuda` |
| torch, float32 (`half=False`) | the same in single precision | `Stitcher(lut, 'cuda', half=False)` |
| OpenCV on the CPU | `cv2.remap` per camera with the tables in OpenCV's fixed-point form, as the runtime does | `--device cpu`, or no GPU |

The result is a (3, H, W) image (a tensor on the device, or a numpy array), the owner map (which camera each pixel
shows) and a validity mask.

## Measured (tests/test_bevstitch.py and `falcon bev bench`; RTX 5080, torch 2.9, 2026-11-12)

- **Geometry**: four 190° equidistant fisheyes (480 × 300) on the MKZ, the ground rendered into each through the lens
  model's inverse with its own coordinates as the pixel values, stitched back at 4 cm per pixel: every covered pixel
  (away from a camera's image border and image circle) holds its own position to **under 1 pixel (max), 0.3 px at the
  99th percentile, 0.05 px median** — the tables, OpenCV's remap and the seam feathering together.
- **A checkerboard** stitched the same way matches the true squares away from their edges (mean difference under 1.5
  of 255; the edges blur by each camera's own pixel footprint, several centimetres far from a camera).
- **torch against OpenCV** on the CPU: mean difference 0.35 of 255, at most 4 (OpenCV quantises sub-pixel positions to
  1/32 px). **GPU float32 against OpenCV**: the same. **GPU half precision**: mean under 1 level, the sampling grid's
  11-bit mantissa moving a sample by up to a quarter pixel of a 480-pixel image (at 1080p, 0.47 BEV pixels at the 99th
  percentile against 0.03 in float32, on the coordinate test).
- **Speed**, four 1920 × 1080 fisheyes → 485 × 645 px at 2 cm: **122–137 fps** with the frames arriving in host
  memory (their upload timed: the bound), 129 fps in float32; **~1900 fps** with the frames already on the GPU; peak GPU
  memory 0.35 GiB. The acceptance target (≥ 30 Hz for 4 × 1080p fisheyes) is met with room. OpenCV on the CPU: tens of
  fps at that size (`falcon bev bench --device cpu`).
- **The runtime**: the exported tables stitched by the bundled `stitcher/bev_stitch.py` match Falcon's OpenCV stitcher to
  within 2 levels (the runtime's weights are 8-bit), at the frames' own resolution or half of it.

## The exported tables and the runtime on the vehicle

`falcon bev export` (and *Export tables…*) writes the layout Vantage's export writes and its runtime reads:
`bev_lut.json` (the frame, extent, image axes, cameras with their poses, seams, body and limits), `bev_lut.npz`
(`<camera>_map` H × W × 2 float32, `<camera>_weight` uint8, `body_mask`, `covered`), `bev_lut.txt` (the C++ reference
stitcher's manifest), per-camera `.bin` files, and `stitcher/` with Vantage's reference stitcher `bev_stitch.py` and its
licence. The stitcher is **Apache-2.0** (unlike the rest of Falcon, GPL-3.0-or-later): it is copied unchanged from
Vantage into `falcon_autonomy/runtime/` (`bev_stitch.py` with its SPDX header, `LICENSE-APACHE-2.0`), and needs only
numpy and OpenCV — on a Jetson AGX Orin its `--cuda` remaps on the GPU when OpenCV was built with CUDA:

```sh
python3 stitcher/bev_stitch.py --frames captures/ --out bev.png --balance          # captures/<camera name>.png
python3 stitcher/bev_stitch.py --source front=/dev/video0 --source rear=/dev/video1 \
    --source left=/dev/video2 --source right=/dev/video3 --balance --show --cuda
```

## Rigs ▸ Surround view

The selected rig's recorded runs (or any run chosen), a frame slider, the seams drawn over the view, exposure balancing,
OpenCV or CUDA, the tables' size and coverage, the seams and their source, the frame time, and *Export tables…*.

## Limits

- **The ground plane only**: a bird's-eye view projects every pixel onto the flat ground, so anything above it — cars,
  kerbs, people, poles — is smeared outwards along each camera's rays, differently by each camera, and doubles or
  vanishes at a seam. A 3D bowl (Vantage's `core/bowl.py`) is not built here yet.
- The ground is taken as flat under the vehicle's origin: a slope, a camber, the body's pitch under braking, or a mount
  that moved (a rig's mount faults) misalign the cameras on the ground.
- The body is a box; a Vantage project's body box when the rig came from Vantage. Mirrors are not modelled as occluders.
- Seams estimated here use the box as the only occluder (Vantage uses the body's mesh); a Vantage rig's seams are used
  as they are.
- Measured on synthetic ground and on this machine's GPU; the Orin's speed with the runtime is not measured here (it
  needs the board: the `orin` marker).
