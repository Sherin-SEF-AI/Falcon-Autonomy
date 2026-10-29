# Sim-to-real image translation

`falcon translate` (Training ▸ *Sim-to-real translation*) learns to make Falcon's rendered camera frames look like a
real dataset's, translates recorded drives into new runs **whose every label stays valid**, measures how far each
translated frame moved from its input, and measures the gap — and the downstream effect — after. Code:
`falcon_autonomy/core/translate.py` and `falcon_autonomy/core/translate_net.py` (their docstrings are the reference).

The [measured gap](sim-to-real.md) says why: CARLA against KITTI is FD 184.8 against a real-vs-real floor of 23.1,
and global appearance matching does not reduce it — the gap is content and rendering, not colour. A learned
translator changes local texture and rendering, which matching cannot. Whether that helps a detector is a separate
question, answered by an experiment, not assumed.

## The method

**CUT** — contrastive unpaired translation (Park, Efros, Zhang and Zhu, *Contrastive Learning for Unpaired
Image-to-Image Translation*, ECCV 2020) — in Falcon's own implementation (GPL-3.0-or-later; the architecture follows
the paper and its BSD-licensed reference implementation's design; no code was copied):

| part | what |
|---|---|
| generator | ResNet generator (Johnson et al. 2016): 7 × 7 stem, two stride-2 downsamplings, 9 residual blocks, two upsamplings, InstanceNorm |
| discriminator | 70 × 70 PatchGAN (Isola et al. 2017), least-squares loss (Mao et al. 2017) |
| PatchNCE | five encoder taps (pixels, stem, both downsamplings, the middle residual block), 256 random patches each, a two-layer projection head per tap, InfoNCE at τ 0.07 with negatives from the same image: each output patch must pick the input patch at the same place |
| identity | PatchNCE between a real image and the generator's output on it (averaged with the main NCE, as CUT does) |

**Geometry-preserving by design.** No layer can move a pixel: no flow, warping, deformable or attention layers and
no global pooling; receptive fields are local. Upsampling is nearest-neighbour + convolution, not transposed
convolution (whose uneven overlap prints a checkerboard of false edges; Odena et al. 2016). The output is a residual
in the logit of the input, `tanh(artanh(x) + head(x))`, with the last convolution initialised to zero: the untrained
generator is exactly the identity, and training moves appearance away from it.

**The label as a conditioning input.** The generator only ever translates simulated frames, which always carry
Falcon's semantic label, so it is given the frame's label as ten one-hot channels (road, sidewalk, structure,
pole/sign, vegetation, terrain, sky, person, vehicle, other — CARLA's 29 tags grouped). It need not guess which pixels
are road or car. A real image in the identity pass gets all-zero channels ("unknown"). `--no-condition` turns this off.

**Not implemented, and why.**
- *A semantic-consistency loss from a frozen segmenter on the output* (CyCADA, Hoffman et al. 2018): it needs a
  segmenter trained on real data, which Falcon does not download. The label as an input is the form used.
- *CycleGAN* (Zhu et al. 2017): two generators and two discriminators double the cost, and cycle consistency lets a
  generator hide its input in imperceptible signal instead of keeping it in place (Chu, Zhmoginov and Sandler 2017,
  *CycleGAN, a master of steganography*). The one-sided PatchNCE ties output to input where it is.
- *Diffusion with ControlNet*: not implemented — it needs a model download whose licence the user must accept.

## Data and terms

```sh
falcon translate train --workspace W --id kitti-cut --sim RUN [--sim RUN…] --real kitti_raw:/data/kitti/2011_09_26 \
    [--iterations 12000] [--lambda-nce 10] [--batch 4] [--crop 192] [--seed 0] [--no-condition] [--no-idt]
```

- **Simulated**: Falcon runs, one pinhole camera each (`RUN?camera=C&every=N`), with the semantic label.
- **Real**: any [gap](sim-to-real.md) set read in place — `kitti_raw`, `kitti_object`, `nuscenes`, `bdd100k`, `coco`,
  `folder`, `dashcam` (your own footage, [dashcam](dashcam.md)) (`KIND:PATH?limit=N`). Nothing is copied into the project or the translator's folder (`data.json` lists
  the paths).
- Every image is resized to 640 px wide (the gap's working width); training takes seeded random crops of 192 px
  (clamped to the smallest image: KITTI is 193 px tall at that width) with a seeded horizontal flip.

**A translator inherits the terms of its real data, and so does every image it makes.** KITTI is CC BY-NC-SA 3.0
(attribution, non-commercial, share-alike): a translator trained on KITTI images, and the translated runs it makes,
may not be used commercially. `translator.json` (`terms`, `licence_note`), every translated run's `run.json` and the
page say so. nuScenes, BDD100K and COCO carry their own terms (as their owners publish them; check the current
terms); a plain folder's terms are unknown to Falcon.

## Training

Adam (2·10⁻⁴, β 0.5, 0.999) for the generator with its projection heads and for the discriminator; constant, then
linearly to zero over the second half; λ_GAN 1, **λ_NCE 10** (`--lambda-nce`; CUT's default is 1, FastCUT's 10 —
measured below, λ 1 repainted content and every held-out frame exceeded the drift threshold); 12 000 iterations;
bfloat16 autocast on CUDA; batch 4. Seeded: crops, flips and
patch draws are functions of (seed, iteration). **On the CPU two runs, and a run cancelled and resumed, give
bit-for-bit the same weights (tested); on CUDA cuDNN's kernels are not guaranteed to.** A checkpoint every 1000
iterations and on a cancel (SIGTERM, Ctrl+C, `{"cmd": "cancel"}`); the same command resumes; a finished translator is
not trained again, and a folder made with other settings or data is refused. A finished training keeps no optimizer
state (`--keep-checkpoint` keeps it).

**GPU.** Training is a `needs_gpu` job: the job queue never runs it while a CARLA job holds the GPU; run from the CLI on
CUDA it takes the CARLA and GPU locks itself, as `falcon train` does ([training](training.md#gpu-sharing)). The
process's share of GPU memory is capped at 0.6.

**Every 1000 iterations** eight fixed simulated frames are translated whole and their drift measured (below):
`metrics.jsonl`'s `probe` is the training curve of label preservation, beside the losses and images/s.

```
translators/<id>/
  translator.json   the model card: method and paper, architecture, settings, data (runs, counts, real sets, a
                    SHA-256 over the data list), terms and licence note, the weights' SHA-256, measured cost
                    (seconds, images/s, peak GPU memory), the last probe
  run.json          provenance (inputs hashed, versions, device), timings, outputs' hashes
  generator.pt      the generator's weights (checked against translator.json before every use)
  data.json         every image used (paths, not copies)
  metrics.jsonl     per 100 iterations: losses, learning rate, images/s; per 1000: the probe drift
  previews/         four probe frames, before (top) and after (bottom)
```

## Translating a drive

```sh
falcon translate apply --model W/translators/kitti-cut --run RUN [--run RUN…] --workspace W \
    [--threshold 0.35] [--on-drift raw|flag] [--camera C] [--native] [--copy]
```

A new run per drive, `W/runs/<drive>-tr-<translator>`:

- every per-frame file of the source — semantic, instance, range, objects, flow, lidar and its labels, radar — is
  **hard-linked** (the same bytes, no space; copied across file systems or with `--copy`), and the run's own records
  (frames.jsonl, actors.jsonl, lanes.json, …) are copied. Falcon's tools write by replacing files, never in place, so
  a hard link cannot carry a change back to the source; a tool of your own that edits a per-frame file in place would
  change both;
- each pinhole camera's rgb is translated and written at **its own resolution**: the network runs at 640 px wide and
  its change (output − input) is upsampled (bicubic) and added to the full-resolution input, so the labels' pixel grid
  is the source's. `--native` runs the network at the camera's full resolution instead (it was trained at 640);
- `run.json` keeps the source's (its start time, rig, settings and faults, so timestamps and calibration are the
  source's) with `worker: translate` and a `translation` block: the source and its run.json's SHA-256, the model and
  its weights' SHA-256, the settings, the drift summary, the terms; `translation.jsonl` holds every frame's measures.

Exporters, the harness, `falcon gap`, `falcon train` and the experiments read it as any drive (tested: the same boxes,
classes and ignore flags as the source; a COCO export). The Data, Models and Training pages list it with the drives.
`falcon reproduce` does not apply (it re-runs a drive).

## Label preservation, measured

Per translated frame, at 640 px wide, on grayscale:

| measure | what |
|---|---|
| structure | the mean of SSIM's structure term (Wang et al. 2004), (σxy + C3) / (σx σy + C3), Gaussian window σ 1.5 px — insensitive to brightness and contrast, falls when edges move or vanish |
| edge recall ("lost structure") | the input's strong edges (Canny, thresholds at its own top 5 % of gradient magnitude) found within 1.5 px among the output's edges (top 15 %) |
| edge precision ("added structure") | the output's strong edges found within 1.5 px among the input's |
| worst object box | each labelled object (car … pedestrian, at least 16 px tall at this width): its box's mean structure; the frame keeps the worst |
| light recall, light precision | night frames only (median grey < 60): the input's light sources found in the output, and the output's in the input ([at night](#at-night)) |
| **drift** | 1 − min(structure, edge recall, edge precision, worst object box[, light recall, light precision]) |

Thresholds come from each image's own gradients, so a tone change alone does not move the edges.

**The threshold is a heuristic** (0.35), calibrated on transforms of known size applied to ten real CARLA frames
(Town10, 1280 × 720, measured at 640 px):

| transform | structure | worst box | edge recall | edge precision | drift |
|---|---|---|---|---|---|
| identity | 1.000 | 1.000 | 1.000 | 1.000 | 0.00 |
| JPEG quality 75 | 0.959 | 0.962 | 0.999 | 0.999 | 0.04 |
| Falcon's appearance matching (to KITTI) | 0.985 | 0.966 | 0.942 | 0.987 | 0.06 |
| Gaussian noise σ 4 | 0.969 | 0.969 | 1.000 | 1.000 | 0.03 |
| gamma 0.6 | 0.998 | 0.985 | 0.997 | 0.980 | 0.02 |
| Gaussian blur σ 1.5 px | 0.877 | 0.774 | 0.936 | 0.998 | 0.23 |
| whole image shifted 1 px | 0.811 | 0.747 | 1.000 | 1.000 | 0.25 |
| whole image shifted 2 px | 0.629 | 0.479 | 0.805 | 0.803 | 0.52 |
| smooth warp, 2 px amplitude | 0.603 | 0.205 | 0.786 | 0.784 | 0.80 |

0.35 lies between a 1-px and a 2-px shift of the whole frame. A frame over it is **refused** by default — its raw
image is kept in the translated run (`--on-drift raw`) — or kept and listed (`--on-drift flag`). The structure term
also falls when texture is replaced inside an object (a GAN's hallucination), which is what the worst-box measure is
for. It does not tell a "realistic" texture change from a wrong one: a drift under the threshold is not proof that a
human would label the frame the same.

### At night

Dark frames hide changes from the structure term: where a frame's local contrast is a few grey levels, SSIM's
constant C3 (29 in squared 8-bit levels) dominates and the term reads ≈ 1 whatever happens there, and the edge maps
follow each frame's own strongest gradients — the lights. Measured on ten frames of each set (640 × 360, as
rendered or extracted), the same transforms as above:

| transform | night dashcam | night Town10HD | night Bengaluru (imported) | day Town10HD, 640 px rendered |
|---|---|---|---|---|
| JPEG quality 75 | 0.02 | 0.08 | 0.10 | 0.03 |
| Gaussian noise σ 4 | 0.08 | 0.09 | 0.11 | 0.07 |
| gamma 0.6 | 0.00 | 0.01 | 0.12 | 0.00 |
| brightness × 2 | 0.11 | 0.00 | 0.01 | 0.52 |
| Gaussian blur σ 1.5 px | 0.08 | 0.09 | 0.25 | 0.09 |
| whole image shifted 1 px | 0.07 | 0.25 | 0.28 | 0.12 |
| whole image shifted 2 px | 0.16 | 0.34 | 0.28 | 0.26 |
| smooth warp, 2 px amplitude | 0.05 | 0.13 | 0.12 | 0.08 |
| twelve lamps painted into the dark areas | **0.21** (0.04 without the light terms) | **0.59** (0.03) | **1.00** (0.01) | 0.01 |
| every light source removed (pixels > 200 and a 4 px ring set to the median) | **0.90** (0.27) | **0.86** (0.03) | — (no light > 200) | 0.77 |

Geometry registers about as by day (on the real frames a 2 px shift reads 0.16 at night, 0.17 by day; frames
rendered at 640 px are softer than the 1280 px frames of the table above, so their shifts read lower there too).
**Lights did not**: removing every light from a night frame read 0.03–0.27 and painting twelve lamps into the dark
0.01–0.04 — both under the threshold, so a translator could add or remove light sources (headlights included)
unseen. On a **night frame** (its median grey below 60) the drift therefore adds two terms:

- **light recall**: the input's light sources (blobs of at least 4 px at grey ≥ 200) found in the output (grey ≥ 100
  within 4 px) — lights removed;
- **light precision**: the output's light sources found in the input, the same way — lights invented.

Grey ≥ 100 lets a light dim or brighten by a stop: with 150 instead, doubling a night Town10 frame's brightness read
0.81 (its lit walls crossed 200 and counted as invented lights); with 100, 0.00. Daytime frames are measured as before
(no light terms). The threshold stays 0.35: on the night dashcam frames twelve invented lamps (0.21) still pass —
they are a small share of the dozens of lights those frames hold — while on the simulated frames, where lights are
few, they are refused. The terms are heuristics with fixed grey levels, checked on these four sets only.
Translating Town10HD at night towards the user's night footage, the light terms refused 31 of 600 frames (28 invented
lights, 3 removed); the frames they keep can still carry glare spread over road the source already lit
([dashcam, at night](dashcam.md#at-night-1)).

`falcon translate check RUN… [--real KIND:PATH [--features --workspace W]] [--segmenter MODEL.json] [--out DIR]`
recomputes every frame's drift against its source (refused if the source's run.json changed), and with `--real`
measures the gap of the source frames and of the translated frames against the real set — the same frames on both
sides, the [gap](sim-to-real.md)'s statistics, FD and KD with their real-vs-real floor. With a registered segmenter
(any [model](models.md) of kind segmenter) it adds the pixel agreement of its output on the raw and translated
frames and its mIoU against the labels on both — none is bundled, so none was measured here.

## Measured

On the reference machine (RTX 5080, 16 GB; torch 2.9.1 + CUDA 12.8), 2026-10-29. **Training data:** 400 frames of
two real CARLA 0.9.16 Town10HD recordings (1280 × 720 with labels) and 451 KITTI raw images of drives 0005 and 0015
(2011_09_26, left colour camera). **Held out:** the two drives of the [gap measurement](sim-to-real.md#measured-carla-against-kitti)
(Town10HD and the imported central Bengaluru, 240 frames) on the simulated side, and KITTI drive 0027 (188 images) on
the real side. Two settings, each trained once (seed 0):

| | A: λ_NCE 1 (CUT's default) | B: λ_NCE 10 (FastCUT's weight) |
|---|---|---|
| iterations (batch 4, 192 × 192 crops) | 20 000 | 12 000 |
| time | 1722 s | 1147 s |
| images/s (median over the run) | 47.1 | 45.9 |
| peak GPU memory allocated | 1931 MiB | 1927 MiB |
| probe drift, median of 8 Town10 frames, every 1000 iterations | 0.18 0.04 0.13 0.36 0.30 0.43 0.24 0.31 0.28 0.27 0.37 0.46 0.50 0.42 0.32 0.52 0.44 0.40 0.51 **0.49** | 0.03 0.00 0.00 0.01 0.04 0.05 0.06 0.11 0.07 0.10 0.13 **0.15** |

**Drift on the held-out frames** (medians; drift threshold 0.35):

| | structure | edge recall | edge precision | worst object box | drift | over the threshold |
|---|---|---|---|---|---|---|
| A, Town10HD | 0.808 | 0.569 | 0.770 | 0.691 | 0.432 | 120 of 120 |
| A, Bengaluru | 0.843 | 0.558 | 0.431 | — (no objects) | 0.570 | 120 of 120 |
| B, Town10HD | 0.935 | 0.888 | 0.918 | 0.908 | 0.113 | **0 of 120** |
| B, Bengaluru | 0.952 | 0.747 | 0.537 | — (no objects) | 0.463 | 119 of 120 |

Setting A changes too much: by its last iteration it repaints buildings as trees and loses half the strong edges, and
its probe drift climbed through training — **every held-out frame is refused**. Setting B keeps the structure of the
Town10HD frames (all translated). Bengaluru is a generated world of flat, untextured ground, unlike anything in the
training frames: B paints structure onto it (edge precision 0.54 — "added structure") and 119 of its 120 frames are
refused. The check catches what the eye sees in the previews; it is what makes a translated run usable at all.

**The gap after translation** (`falcon translate check … --features`, the gap's method, seed 0; the raw rows
reproduce [the documented gap](sim-to-real.md#measured-carla-against-kitti) exactly, FD 184.8):

| simulated frames | real set | FD raw → translated (floor) | KD raw → translated (floor) |
|---|---|---|---|
| Town10HD + Bengaluru, B (Bengaluru refused: raw) | KITTI, three drives (two seen in training) | 184.8 → 181.0 (23.1) | 0.113 → 0.100 (−0.001) |
| the same | KITTI drive 0027 (held out) | 244.6 → 235.8 (19.0) | 0.240 → 0.223 (0.000) |
| Town10HD only, B | KITTI, three drives | 159.7 → 152.1 (23.1) | 0.137 → 0.128 (−0.001) |
| Town10HD only, B | KITTI drive 0027 (held out) | 248.4 → 231.3 (19.0) | 0.318 → 0.290 (0.000) |
| Town10HD + Bengaluru, A kept despite the drift (`--on-drift flag`) | KITTI, three drives | 184.8 → 191.6 (23.1) | 0.113 → 0.098 (−0.001) |
| the same | KITTI drive 0027 (held out) | 244.6 → 222.5 (19.0) | 0.240 → 0.186 (0.000) |

Appearance statistics move most: on Town10HD with B, CIELAB lightness 48.2 → 35.1 (KITTI 38.8; shift 1.63 → 0.97) and
contrast 23.3 → 29.9 (KITTI 29.0; shift 1.60 → 0.57). **The feature distances fall by 5–7 % with B — far from the
real-vs-real floor.** A falls further on the held-out drive (−9 %) by repainting content, which is exactly what destroys
the labels. What remains of the gap is what a label-preserving translation must not change: the scenes themselves
(Karlsruhe in 2011 against CARLA's town), their geometry and their objects. These are single trainings of one seed:
the training noise of the numbers is not measured.

**Cost of applying**: 120 frames of 1280 × 720 in 69–80 s on the CPU (the network at 640 × 360, the full-resolution
composite, the drift measures and linking the labels), about 1.6 frames/s (the GPU was busy; not measured there).

## The downstream experiment

```sh
falcon translate experiment --workspace W --id ID --synthetic falcon:RUN… --translated falcon:RUN_TR… \
    --real-train KIND:PATH --real-test KIND:PATH [--shares 50] [--budget N] [--seeds 3] [--epochs E] [--arch A]
falcon translate results W/experiments/ID
```

The [mixed experiment](training.md#the-mixed-simreal-experiment)'s design with three kinds of arm: **real only**,
**raw synthetic** and **translated synthetic** at the same share of the same budget. A raw and a translated arm at one
share and seed take **the very same frames** (one seeded permutation; each translated run is matched to its source
frame by frame, and refused if its labels differ), so they differ only in the pixels. Every run trains for the same
iterations and is scored on the same held-out real test set (leakage refused). Results: each arm's mean AP over seeds
with a bootstrap interval over the test set's units; **translated − raw**, and each against real only, as paired
differences with the combined interval over test-sample and seed noise; "better"/"worse" only when that interval
excludes zero. `results.json` (`falcon.translation_results/1`); the Training page's experiment view shows it.

**Not measured in the translator's own domain.** No labelled KITTI (or other driving) set is on the reference machine
— KITTI raw has no 2D labels on disk — so whether KITTI-styled synthetic data helps a detector *on KITTI* is not
measured. With a labelled KITTI object set the experiment is ready to run:

```sh
falcon translate experiment --workspace W --id kitti-tvr \
    --synthetic falcon:W/runs/drive --synthetic falcon:W/runs/lead \
    --translated falcon:W/runs/drive-tr-kitti-cut --translated falcon:W/runs/lead-tr-kitti-cut \
    --real-train kitti_object:/data/kitti/training --real-test kitti_object:/data/kitti/testing_split \
    --shares 50 --seeds 3 --epochs 8
```

(the real training and test sets must share no raw drive: the devkit's mapping is used, as in the mixed experiment).

**A proxy, measured: COCO street images.** COCO val2017 is on the machine: its 611 images with a car, truck or bus at
least 24 px tall, split by image id into 313 for real training (47 carved off for validation) and 298 for testing
(2279 objects). Synthetic: the two Town10HD drives (400 frames), raw or translated by setting B (none refused). Budget
240 images, 50 % synthetic, 3 seeds, Faster R-CNN on MobileNetV3-Small from ImageNet weights, 640 × 384, 8 epochs
(240 iterations), seed 0:

| arm | AP [95 % CI] | per seed | AP50 |
|---|---|---|---|
| real only | 0.0283 [0.0236, 0.0358] | 0.0290 0.0287 0.0271 | 0.077 |
| raw 50 % | 0.0123 [0.0101, 0.0168] | 0.0131 0.0103 0.0134 | 0.039 |
| translated 50 % | 0.0102 [0.0084, 0.0136] | 0.0106 0.0090 0.0111 | 0.035 |

Translated − raw: −0.0020 (bootstrap [−0.0038, −0.0008]; with the seed spread [−0.0048, +0.0008]) → **no detectable
difference at 95 %**. Both synthetic arms are worse than real only (−0.016 and −0.018). **This says little**: the
detector is barely trained (AP 0.03 at best — a small backbone, 240 iterations), the test domain (COCO's street
photographs) is not the domain the translator was trained towards (KITTI), and the synthetic frames replace half of
the real ones at a fixed budget. It is reported because it was run; it is not evidence either way about translation
on KITTI.

## Limits

- Images only: lidar, radar and their labels are the source's, untouched (and not made more real).
- One translator per real domain; it learns that domain's look — and its content statistics: on scenes unlike its
  training frames it hallucinates (measured below), which the drift check refuses but cannot repair.
- The drift measures are heuristics of geometry, not of semantics: a car re-textured as a plausible other car passes.
  Semantic agreement needs a segmenter trained on real data (supported, none bundled).
- Trained at 640 px wide; the full-resolution output adds the network's change upsampled, so fine texture above that
  resolution is the source's (`--native` runs at full resolution, outside the training scale).
- The GAN is not bit-for-bit reproducible on CUDA, and its quality varies along training (the probe curve shows it);
  the final iteration is kept, not a selected one.
- Terms: a translator and its images carry the real data's terms (KITTI: non-commercial).
