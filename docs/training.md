# The training lab

`falcon train` fine-tunes a 2D box detector on Falcon drives, dataset versions and real labelled sets, and exports it
as an ONNX model the [harness](models.md) runs. `falcon experiment` tracks runs, exports them for MLflow, and runs the
**mixed sim/real experiment**: real only against real + X % synthetic, at a fixed training budget, with honest
intervals. Code: `falcon_autonomy/core/training.py` and `falcon_autonomy/core/experiments.py`.

## Models and licences

Detectors are built from torchvision (BSD-3-Clause) parts, exactly as torchvision 0.24's own builders build them
(checked: the same state dictionaries, modules and detections). No Ultralytics or other AGPL code.

| arch | what | COCO weights |
|---|---|---|
| `fasterrcnn_mobilenet_v3_large_fpn` | Faster R-CNN, MobileNetV3-Large FPN (the default) | yes |
| `fasterrcnn_mobilenet_v3_small_fpn` | the same design on MobileNetV3-Small, composed by Falcon from torchvision parts | no |
| `fcos_resnet50_fpn` | FCOS, ResNet-50 FPN | yes |
| `retinanet_resnet50_fpn_v2` | RetinaNet v2, ResNet-50 FPN (GroupNorm head, GIoU loss) | yes |

**Initialisation** (`--init`): `scratch` (random weights, trainable BatchNorm), `imagenet` (the backbone's ImageNet-1k
weights; three trainable backbone stages and frozen BatchNorm — RetinaNet v2 keeps plain BatchNorm — as torchvision
builds them) or `coco` (the whole detector's
COCO weights). With COCO weights the class head is cut down to Falcon's classes and **each row starts as the matching
COCO class's** — person → pedestrian, car, truck, bus, motorcycle, bicycle, train, traffic light, stop sign →
traffic_sign — so fine-tuning starts from a detector that already finds them; a class COCO lacks (rider) starts fresh.

**Weights are never downloaded by themselves.** Each file is pinned by URL and full SHA-256; `falcon train --weights`
lists them and `falcon train --fetch NAME --workspace W` downloads one into `W/models/pretrained/` (URL, size, licence
and hash shown first; refused on a mismatch). A file is looked for in `--weights-dir`, the workspace's
`models/pretrained/` and torch's hub cache, and **checked against its SHA-256 before every use** wherever it was
found. The hashes were computed on the files downloaded from download.pytorch.org (2026-10-28); the first eight hex
digits of each match torchvision's file-name suffix.

| weights | file | MiB | SHA-256 |
|---|---|---|---|
| `fasterrcnn_mobilenet_v3_large_fpn-coco` | fasterrcnn_mobilenet_v3_large_fpn-fb6a3cc7.pth | 74.2 | `fb6a3cc7…df3533` |
| `fcos_resnet50_fpn-coco` | fcos_resnet50_fpn_coco-99b0c9b7.pth | 123.6 | `99b0c9b7…b9e7` |
| `retinanet_resnet50_fpn_v2-coco` | retinanet_resnet50_fpn_v2_coco-5905b1c5.pth | 146.0 | `5905b1c5…b769` |
| `mobilenet_v3_large-imagenet1k-v1` | mobilenet_v3_large-8738ca79.pth | 21.1 | `8738ca79…1a04` |
| `mobilenet_v3_small-imagenet1k-v1` | mobilenet_v3_small-047dcff4.pth | 9.8 | `047dcff4…4e1f` |
| `resnet50-imagenet1k-v1` | resnet50-0676ba61.pth | 97.8 | `0676ba61…fb8a` |
| `resnet50-imagenet1k-v2` | resnet50-11ad3fa6.pth | 97.8 | `11ad3fa6…79ca` |

torchvision's code is BSD-3-Clause; the COCO weights were trained on COCO 2017 (annotations CC BY 4.0, images under
their Flickr terms) and the backbones on ImageNet-1k (its own terms). **A model also carries the terms of the data it
was fine-tuned on**: KITTI is CC BY-NC-SA 3.0, nuScenes CC BY-NC-SA 4.0 for non-commercial use, BDD100K has its own
licence (as their owners publish them; check the current terms). The exported registration's `licence_note` lists
what applies; check it before you ship a model.

## Data

A source is `KIND:PATH?key=value&…` (repeat `--train`, `--val`, `--test`):

| kind | what | options |
|---|---|---|
| `falcon` | a recorded drive (needs the `boxes2d` label) | `camera`, `every` |
| `version` | a [dataset version](dataset-format.md) file and one of its splits (`--version V.json` = train on `train`, validate on `val`); a run whose run.json changed since the version was made is refused | `split`, `workspace`, `camera`, `every` |
| `kitti_object`, `coco`, `bdd100k`, `nuscenes` | real sets, read in place as in [sim-to-real](sim-to-real.md) | `labels`, `camera`, `limit`, `seed`, `units` |

Class names are mapped to Falcon's as the gap does (Van → car, Cyclist → bicycle, person → pedestrian, Tram → train, …).
Classes not trained (`--classes`; default car, truck, bus, motorcycle, bicycle, pedestrian, rider) are left out of
training and validation alike, as the harness leaves out classes a model cannot name. Note that real sets name things
differently: KITTI's Cyclist box holds the bicycle and its rider, where Falcon has two classes.

**The ignore rule is the harness's.** A Falcon object with fewer than 20 visible pixels, or more than 90 % of its box
hidden (`core.harness` MIN_PIXELS, MAX_OCCLUDED), is not a training target, and in validation it is neither found
nor missed (a detection on it is not a false positive) — so the trainer optimises what the harness scores (the tests
check, frame by frame, that training's ground truth is the harness's). torchvision's losses have no ignore regions: an
ignored object's pixels count as background in training. Real sets carry no visible-pixel counts: a real box smaller
than 20 px² is ignored (its area bounds its visible pixels); their occlusion is not applied; COCO crowd boxes are left
out by the reader, so they count as background.

Every image is resized to the model's input (`--size`, default 640 × 384) exactly as `core.models.preprocess` does at
inference — a stretch, or with `--letterbox` a letterbox — so training, validation and the exported model see the same
pixels. Augmentation is a horizontal flip (seeded); there is no other.

## Training

SGD with momentum 0.9 and weight decay 10⁻⁴; the learning rate of torchvision's reference recipes (0.02 per 16 images
for Faster R-CNN, 0.01 for FCOS and RetinaNet) scaled linearly with the batch; a linear warm-up, then cosine (or steps
×0.1 at 2/3 and 8/9); gradient clipping at a global norm of 10; automatic mixed precision (float16 with a gradient
scaler) on CUDA. After every epoch: a checkpoint and COCO AP on the validation sources (`core.metrics`, pycocotools'
numbers). The exported model is the epoch with the best validation AP (`--select last` for the last). `--patience N`
stops after N epochs without a better AP. Progress, logs and `val_AP` / `train_loss` metrics go out as
[protocol](protocol.md) lines.

**Cancel and resume.** SIGTERM, Ctrl+C or `{"cmd": "cancel"}` on stdin saves a checkpoint where the training stood —
mid-epoch included — writes run.json with status `cancelled` and exits 3. Running the same command again resumes from
it; the same folder with other settings or data is refused. A finished training is not run again.

## Reproducibility

Seeded: the order and flips of an epoch are functions of (seed, epoch); torch's generator is re-seeded from (seed,
epoch, iteration) before each iteration; cuDNN runs deterministic with benchmarking off, and
`torch.use_deterministic_algorithms(True, warn_only=True)` is on.

- **CPU: bit for bit.** Two runs with the same settings, data and thread count give the same weights, and a run
  cancelled mid-epoch and resumed gives the weights of one that was not (tested).
- **CUDA: depends on the model** (measured on the reference machine, RTX 5080, torch 2.9.1 + CUDA 12.8, AMP on).
  Faster R-CNN is **not** bit for bit: torchvision's RoIAlign backward has no deterministic CUDA implementation (torch
  warns; run.json keeps the warning). Two identical runs reached validation AP 0.2625 and 0.2639 after 180 iterations
  (largest weight difference 0.0023). FCOS and RetinaNet use no such operation and repeated bit for bit (two and three
  runs of 30 iterations, identical weights). That holds on one machine, driver and library version; another GPU or
  cuDNN may choose other kernels.

## GPU sharing

A training is a `needs_gpu` job (`training.job_spec()` builds its JobSpec): the job queue never runs it while a CARLA
job holds the GPU. Run from the CLI on CUDA it also takes the GPU lock itself and waits while any Falcon process's CARLA
job holds the CARLA lock — and holds that lock, so CARLA jobs queue behind the training. The batch is capped at 32 and
the process's share of GPU memory at `gpu_memory_fraction` (0.6 by default: a heuristic that leaves room for the
desktop); the peak memory allocated is measured into run.json.

## Outputs

```
training/<id>/
  run.json          provenance (inputs hashed: drives' run.json, dataset version, real labels files, pretrained
                    weights), settings, seed, versions (torch, torchvision, CUDA, cuDNN, onnx, onnxruntime), device and
                    GPU, measured timings (train and validation seconds, images/s, peak GPU memory), determinism notes,
                    the outputs' hashes
  data.json         every image used: its id, unit, domain (synthetic or real), boxes and ignored boxes; a SHA-256 over
                    the labels; the data's terms
  metrics.jsonl     per epoch: loss and its parts, learning rate, images/s, validation AP, AP50, AP75, AR100 and AP
                    per class
  checkpoints/      last.pt (resume) and best.pt
  model.onnx        opset 17, a fixed input size; loads in onnxruntime 1.28
  model.json        a registration in core.models' format: kind detector, runtime onnx, output boxes_scores_labels
                    (score threshold 0.05, the model's own NMS IoU), its verification
  test_predictions.jsonl   with --test: the chosen model's detections on the test images (harness frames)
```

**The export is checked every time**: the ONNX model through `core.models.Runner` and the torch model, on up to eight
validation images, both decoded by `core.models.decode_detections`; every detection must have a partner of the same
class within 0.5 px and a score within 10⁻³. Measured after fine-tuning from COCO: Faster R-CNN 196 detections, largest
box difference 0.0002 px; FCOS 800, 0.0004 px; RetinaNet 694, 0.003 px. A barely trained model's large box deltas raise
the difference (up to 0.12 px measured), still within the tolerance. Add `model.json` to the project's models to
evaluate the trained model like any other.

## Measured

On the reference machine (RTX 5080, 16 GB; torch 2.9.1 + CUDA 12.8), fine-tuning from COCO at 640 × 384, batch 8, AMP,
four data-loader workers decoding 1280 × 720 PNGs (720 training images, 2 epochs):

| arch | images/s, epoch 2 | peak GPU memory allocated (reserved) |
|---|---|---|
| fasterrcnn_mobilenet_v3_large_fpn | 126–139 (two runs) | 1123 MiB (1994) |
| fcos_resnet50_fpn | 103 | 2757 MiB (2926) |
| retinanet_resnet50_fpn_v2 | 97 | 2388 MiB (2948) |

The first epoch is slower (40–69 images/s: CUDA start-up and the workers' first images). These images were frames of
drives against the test suite's fake CARLA, scaled up: the throughput and memory are real, the AP is not a finding.

## The mixed sim/real experiment

`falcon experiment mixed --id ID --synthetic SRC… --real-train SRC… --real-test SRC… [--shares 0,25,50,100]
[--budget N] [--seeds K]` asks: **does adding synthetic data help on real data, and by how much?**

**Design.**
- **Arms** are synthetic shares of a fixed budget of N training images (default 0, 25, 50, 100 %). Every arm trains
  with the same settings for the same number of iterations (epochs × ⌈N / batch⌉; early stopping is off so no arm
  trains less). The 0 % arm is real only, and every other arm is compared with it.
- **K seeds per arm** (default 3). Per seed, the real and the synthetic pools are put in a seeded order shared by every
  arm, and an arm takes the first N × (1 − share) real and N × share synthetic images of them: arms differ only in
  the share.
- **Validation** is on real images: the `--real-val` sources, or a `val_share` (default 15 %) of the real training
  pool carved off by sequence or image and kept out of every arm. It picks each run's epoch (with `select: last`,
  nothing is picked).
- **Every arm and seed is scored on the same held-out real test set**, which must share no image and no sequence with
  any training or validation data — otherwise the experiment is refused.

**Honest intervals.**
- An arm's number is the **mean over its seeds** of COCO AP (and AP50, AP75, AR100) on the real test set.
- Its 95 % interval is a **bootstrap over the test set's units**, all arms and seeds resampled together. The unit is
  the image — unless the set says which sequence an image comes from: KITTI object images through the devkit's
  `train_rand.txt` and `train_mapping.txt` (found beside the set or in `devkit/mapping/`) map to their raw drives,
  nuScenes images to their logs, `units=parent` takes a folder per sequence, `units=map:FILE` a JSON map. Then whole
  sequences are resampled, because frames of one sequence are not independent (the harness does the same with drives).
  KITTI object images are frames of a few dozen raw drives: without the mapping the interval is too narrow.
- Each arm against real only: a **paired bootstrap** of the difference on the same resamples, so what the test images
  share cancels out.
- These intervals cover the **test sample only, not training noise**. So the **seed-to-seed spread** is reported
  beside them (per-seed numbers, standard deviation, standard error), and the **verdict** uses a combined interval,
  difference ± 1.96 √(sd_boot² + se_arm² + se_real²) — a normal approximation that treats the two noises as
  independent. "Better" or "worse" only when that interval excludes zero; otherwise "no detectable difference at 95 %",
  which is not "no difference". With one seed the training noise is not measured, and the verdict says so; with fewer
  than five test units the intervals are flagged as not reliable.

**Resumable.** Each arm × seed is a run of the experiment; a finished run is skipped, a cancelled one resumes inside
its training, and the analysis is recomputed only when a run's predictions changed. The definition is fixed: the same
id with another definition is refused (settings that do not change results, such as the number of workers, may
differ). `falcon experiment results DIR` prints the table; `results.json` holds everything, with its method.

## Tracking and MLflow

An experiment is a folder `experiments/<id>/`: `experiment.json` (the definition and its SHA-256) and per run
`runs/<run>/` with `run.json` (status, times, provenance), `params.json`, `metrics.jsonl` (key, value, step, timestamp),
`tags.json` and `artifacts.json` (paths, SHA-256, sizes). `falcon experiment list DIR` shows the runs, `falcon
experiment best DIR --metric test.AP` the best finished one (`--min` when lower is better).

`falcon experiment export DIR --mlflow MLRUNS` writes the experiment in **MLflow's file-store layout** — the layout of
`mlflow.store.tracking.file_store.FileStore` in MLflow 2.x (as its source reads and writes it, 2.0–2.17) — without
MLflow installed: `<experiment_id>/meta.yaml`, per run `<run_id>/meta.yaml` (status 3 finished, 4 failed, 5 killed),
`params/`, `metrics/` (lines `timestamp value step`), `tags/` (with `mlflow.runName`) and `artifacts/` (copies; omit
with `--no-artifacts`). Then `mlflow ui --backend-store-uri file:///…/MLRUNS` shows it. Ids are stable, so an export can be
repeated into the same folder. Names MLflow refuses have their other characters replaced by `_` (for example
`AP.range_m[0,20)` → `AP.range_m_0_20_`); parameter values are cut at 6000 characters and tags at 5000. The layout is
checked structurally by the tests; MLflow itself was not run against it.

## Limits

- 2D boxes only: no segmenter, instance masks or 3D detection in this release.
- One fixed input size per model (the ONNX export's); no multi-GPU training.
- Augmentation is a horizontal flip only; no photometric augmentation or appearance matching during training.
- Real sets' occlusion is not applied to the ignore rule, and COCO crowd regions count as background.
- The mixed experiment's verdict is a normal approximation over two noise sources; with few seeds the seed spread
  itself is uncertain (three seeds give a rough standard deviation).
- Throughput was measured on one machine; the numbers above are this machine's, not a target device's.
