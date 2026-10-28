# Models and the test harness

The Models page (and `falcon evaluate`) scores a perception model on recorded drives against Falcon's own ground
truth, says where it fails, and tells a real regression from noise.

## Registering a model

Models ▸ Register… adds a model to the project (`models` in the project file). A registration names:

| field | meaning |
|---|---|
| kind | `detector` (2D boxes) or `segmenter` (a class per pixel) |
| runtime | `onnx` (onnxruntime: CUDA when `onnxruntime-gpu` is installed, else the CPU), `torchscript` (PyTorch, CUDA when available), `python` (a plug-in, `module:Class`) |
| file | the model file; a relative path is in the workspace's `models/` folder |
| input | width, height, layout (NCHW / NHWC), scale, mean, std, RGB or BGR, letterbox |
| output | `boxes_scores_labels`, `yolo` (rows of cx cy w h + class scores, the layout YOLOv5/v8-style exports write — decoded by Falcon's own code) or `logits` (a segmenter); score threshold and NMS IoU |
| classes, class map | the model's class names in index order, and which Falcon class each one is |
| licence | its SPDX identifier; **an AGPL-licensed model is refused** (its licence would bind the app) |
| source | a pinned download URL and its SHA-256 |

**No model is bundled and nothing downloads by itself.** *Download…* shows the URL, the licence and the hash and asks
first; the file is refused if its SHA-256 differs (`falcon model fetch PROJECT ID` does the same).

A plug-in (`runtime: python`) is any class with `predict(rgb, meta)` returning, per image, a list of
`{'bbox': [x0, y0, x1, y1], 'class', 'score'}` in Falcon's classes (a detector) or an H × W map of Falcon class ids
(a segmenter). `meta` carries the frame's run, step and weather — and its ground-truth objects, which only test
models may use (see below).

**The synthetic detector** (*Add the synthetic detector*) is **not a model**. It reads the ground truth and detects
each object with a probability that falls for small, occluded and fogged objects (a documented formula, seeded), jitters
the boxes, sometimes names the wrong class and adds false positives. It exists to test the harness and to give the
rare-case search a model whose weakness is known. Every evaluation made with it says so.

## What an evaluation does

1. **Frames.** Every n-th frame of one camera of each chosen drive goes through the model (the first pinhole camera
   unless another is chosen). Per frame Falcon keeps the predictions, the ground-truth objects in the classes the
   model can name, the frame's conditions (the drive's weather parameters, world and scenario values) and the model's
   latency, in `predictions.jsonl` — so a comparison never runs a model again.
2. **What is scored.** A ground-truth object with fewer than 20 visible pixels, or more than 90 % of its box hidden, is
   *ignored* (as COCO ignores crowds): neither found nor missed, and a detection on it is not a false positive. The
   numbers are in every evaluation's `method`.
3. **Numbers.** COCO AP (IoU .50:.05:.95), AP50, AP75 and AR100, per class, and AP broken down by range (0–20–40–60–80 m),
   occlusion (0–0.1–0.5), truncation (0–0.15–0.3–0.5) and box height (0–25–40–80–160 px). These are pycocotools'
   numbers exactly (`falcon_autonomy.core.metrics`, cross-checked against pycocotools on 250 images: maximum
   difference 0.0). A breakdown applies COCO's area-range rule to the attribute: objects outside the bin are ignored,
   and so is a detection matched to one. Segmenters: IoU per class and mIoU over the classes the model can name
   (other classes' pixels are ignored).
4. **Confidence intervals.** Every number has a 95 % bootstrap interval over **whole drives**: frames of one drive are
   not independent, and resampling frames would make the interval far too narrow (measured in the metrics tests: the
   drive-level interval is more than 3× wider on correlated frames). With a single drive the interval is over frames,
   and with fewer than five drives the evaluation says its interval is not reliable: a bootstrap over two drives has
   three distinct resamples, and its interval collapses onto the estimate. A breakdown bin with nothing to score shows
   "—".
5. **Conditions.** With two conditions chosen (say fog density and sun altitude), AP per cell of their grid: one cell
   per value for a sweep (few distinct values), else quantile bins. Each cell shows its frames and objects. A cell
   with fewer than 10 objects is grey and gets no colour: a number from three objects is not a finding.
6. **Failures.** The frames with the most errors, worst first: missed objects (red) and false positives scoring 0.5 or
   more (orange), matched per class at IoU 0.5.
7. **Latency** is measured here, per frame (pre-processing, inference, decoding), with the runtime's providers (for
   example `CUDAExecutionProvider`). It is this machine's latency, not a target device's (Jetson: milestone 7).

An ONNX or TorchScript evaluation takes the GPU lock (it never runs while CARLA holds the GPU); a plug-in does not.

## Comparing two evaluations

Models ▸ Compare (or `falcon evaluate --compare A B`) puts B against A on the frames both scored. Every number's
difference B − A gets a **paired** bootstrap interval over drives: both evaluations are resampled together, so the
drive-to-drive variation that both share cancels out. That interval is far narrower than comparing two separate
intervals (under 0.3× in the metrics tests). A **regression** is a number whose interval of B − A lies wholly below
zero; an **improvement**, wholly above. Anything else is "no detectable change at 95 %", not "no change".

## Files

```
evaluations/<time>-<model>/
  evaluation.json     model, drives, method, scores (table with CIs, full COCO output), grid, gallery, latency,
                      providers, provenance (the drives' run.json hashes, the model file's hash, the settings)
  predictions.jsonl   one line per frame: image_id, drive, step, image, gt, det, conditions
comparisons/<a>-vs-<b>/comparison.json
```

## Measured

On three 1 s drives against the fake CARLA (tests): the ground truth itself as a detector scores AP 1.000 with the
interval [1.000, 1.000]. The synthetic detector scores AP 0.111 [0.077, 0.152], falling to 0.027 at 40–60 m and 0 beyond
60 m, as its planted size weakness predicts. The paired comparison flags it as a regression against the ground truth.

On a real CARLA 0.9.16 server (2026-10-28): an 8 s Town10HD drive (1280 × 720 camera with every label and optical
flow, a 32-channel lidar, 40 vehicles, 20 walkers) and a lead_brake scenario run, 400 frames, 1157 scored objects
(103 ignored). The synthetic detector's AP falls with range as planted: 0.60 at 0–20 m, 0.39 at 20–40 m, 0.18 at
40–60 m, 0.07 beyond 80 m; and with height, 0.13 below 25 px against 0.70 above 160 px. Two drives are too few for
the drive-level interval, and the evaluation says so. The evaluation took 17 s; the plug-in's latency is a median of
0.06 ms per frame, measured, though meaningless for a real model.

## Limits

- Boxes only: no instance masks, keypoints or 3D detection scoring in the harness yet (nuScenes NDS is in
  `core.metrics` for 3D detectors; the harness does not run 3D detectors in this release).
- One camera per evaluation.
- Ground truth is CARLA's: what CARLA does not render (for example a generated world's road mesh, which its label
  cameras skip — relabelled from the map) is as good as Falcon's relabelling.
