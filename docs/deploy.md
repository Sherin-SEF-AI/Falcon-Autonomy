# Deploying to the Jetson: TensorRT engines, profiled and gated

Models ▸ Deploy (or `falcon deploy`) turns a registered ONNX detector into TensorRT engines **on the Jetson AGX Orin**
(JetPack 6: L4T r36, TensorRT 10, CUDA 12, Python 3.10) — FP32, FP16 and INT8, optionally on a DLA core — and gates
the model's release on three things measured **on the device**: its accuracy against the host's ONNX model on Falcon's
evaluation frames, its latency, and its energy per frame.

Engines are specific to the GPU and the TensorRT version that built them, so everything that touches an engine runs on
the device: Falcon copies the model and a small tool there over SSH, runs the tool, and brings the results back.

The code: `core/deploy.py` (the host side: frames, stages, statistics, the gate, the report, `falcon deploy`),
`hil/engine_device.py` (the device side, one file), `ui/deploy_tab.py` (the Models page's Deploy tab), `core/models.py`
(the preprocessing and the decoder, copied to the device as `falcon_models.py`).

> **Status.** The whole pipeline runs on the **simulated device** on this machine (below). **Nothing has run on an
> Orin yet** — Falcon's development machine has no Jetson and no TensorRT. The TensorRT backend of the device tool is
> written against TensorRT's documented Python API and trtexec's documented flags and has never executed; the first
> real run is yours (see *What you must do*). No number in this document is the Orin's.

## What runs where

| step | where | how |
|---|---|---|
| choose the frames, resize / letterbox them | this machine | `core.models.geometry()` — the host's own preprocessing — to model-size uint8, stored as `.npz` |
| copy | this machine → device | `scp` with your key (hil.deploy: BatchMode, accept-new host keys, never a password) into `<remote_dir>/deploy/<folder>` |
| INT8 calibration | device | the TensorRT Python API: `IInt8EntropyCalibrator2` over the calibration frames → a calibration cache |
| build | device | `trtexec` → `engine.plan`, the build log, the layer information |
| profile | device | `trtexec` timing + an INA3221 power sampler in the tool |
| inference for accuracy | device | the TensorRT Python API (CUDA memory through `libcudart` by ctypes: no pycuda, no cuda-python); without the Python API, `trtexec` once per frame |
| normalise, decode | device | `falcon_models.py`: a verbatim copy of `core/models.py` (its SHA-256 recorded) — the same code as the host's |
| score, compare, gate, report | this machine | `core.harness` (COCO AP with drive-level intervals), a paired bootstrap, the gate file, `core.reports` |

The device's address, SSH user, key and folder are the ones `falcon hil device` stored in **your settings only**
(`hil_device`); a deploy folder says only "from the user's settings", and messages are cleaned of the address and
the user (the tests check that neither appears in any file written).

## The simulated device

`--device sim` (the default, and *Simulated device* on the page) runs **the same tool, the same files, the same
commands** in a folder on this machine (`DIR/simdevice`), with `--backend onnxruntime`: onnxruntime on the CPU stands
in for TensorRT. Every result it produces is labelled — in `deploy.json` (`simulated: true` and the label), in each
engine's build log and record, on the page, and as the report's first note:

> SIMULATED DEVICE — not TensorRT, not the Orin: onnxruntime on this machine's CPU stands in for the engine; the
> pipeline only. No number here is the Orin's.

Its precisions are stand-ins, not TensorRT's:

| precision | on the simulated device |
|---|---|
| fp32 | the ONNX model as it is |
| fp16 | the ONNX model's float weights rounded to float16, computed in float32 (TensorRT's FP16 kernels also compute activations in FP16: not reproduced) |
| int8 | onnxruntime's static quantisation (QDQ, int8 activations and weights) with **entropy calibration** on the same calibration frames; the "cache" is the quantised model |

There are no INA3221 rails on a PC, so power and energy are **not measured** (never estimated); the DLA does not exist
there. Its gate verdicts are computed (to test the gate) but marked *not a release decision*.

## Stages

Each stage is a command that reads and extends `DIR/deploy.json`; `falcon deploy all` (and the page's button) runs
them in order.

### build

1. **Evaluation frames**: every n-th frame (`--every`) of the chosen camera of each evaluation drive that has object
   labels — exactly the frames `core.harness.collect` scores, so the reference and the engines see the same images.
2. **Calibration frames** (INT8): `--calib-count` frames (default **500**, TensorRT's guidance for entropy calibration)
   spread evenly over the calibration drives (`--calib-run`), **never an evaluation frame**. Calibration drives should
   be other drives than the evaluation's; if they overlap, the evaluation frames are left out and a note says the INT8
   accuracy may be optimistic (neighbouring frames of one drive are alike).
3. **Copy** the bundle: `engine_device.py`, `falcon_models.py`, `model.onnx`, `spec.json` (the input spec, the output
   format, classes and class map, the ONNX input and output names, `--shapes` for a model with dynamic input
   dimensions), `eval/`, `calib/`.
4. **Check** the device (`engine_device.py check`): Python, numpy, TensorRT (Python and `trtexec`), CUDA, L4T, the
   board, the power mode, the clocks, the INA3221 rails, temperatures, DLA cores.
5. **INT8 calibration** (below), cached.
6. **One engine per precision**, and with `--dla N` also `fp16-dlaN` and `int8-dlaN` (the DLA runs FP16 and INT8 only).
   The build log, `build.json` and `layers.json` come back for each.

The exact trtexec command (recorded in `build.json` with its exit code):

```
trtexec --onnx=model.onnx --saveEngine=engines/NAME/engine.plan --memPoolSize=workspace:4096
        --exportLayerInfo=engines/NAME/layers.json --profilingVerbosity=detailed --skipInference --verbose
        [--shapes=INPUT:1x3xHxW]                 a model with dynamic input dimensions
        [--fp16]                                 fp16
        [--int8 --fp16 --calib=cache/int8.cache] int8 (FP16 allowed for layers without an INT8 kernel)
        [--useDLACore=N --allowGPUFallback]      a DLA engine
```

From the build Falcon reads (and keeps the raw log beside): the TensorRT version, the build time, the memory lines,
the **layers by output precision** (`layers.json`'s `Format/Datatype` of each layer's first output: FP32, FP16, INT8,
…), the **DLA subgraphs** (TensorRT names them `{ForeignNode[…]}`) and the **layers that fell back from the DLA to the
GPU** (the log's "Unsupported on DLA … Switching this layer's device type to GPU" lines; the wording differs between
TensorRT versions, so the raw log is kept). The workspace asked for is recorded (`--workspace-mib`, default 4096).

### INT8 calibration

**On the Orin**: `engine_device.py calibrate` uses the **TensorRT Python API** (`python3-libnvinfer`, part of
JetPack's `nvidia-jetpack` meta-package): an `IInt8EntropyCalibrator2` feeds the calibration frames — normalised on the
device by `falcon_models.normalise()` — one at a time (batch 1) while TensorRT builds an INT8 network, and writes the
**calibration cache** `cache/int8.cache`. Every INT8 build (GPU or DLA) then reads it with `trtexec --calib`.
`trtexec` alone cannot calibrate: without a cache it uses placeholder scales, so Falcon **refuses** INT8 when the
Python API is missing rather than build a meaningless engine. The cache is reused when its key — the SHA-256 of the
model, the calibration frames, the input spec and the backend — matches (`cache/int8.cache.key`); `calibrate` records
the frame count, the time and the cache's SHA-256. (TensorRT 10 marks implicit-quantisation calibrators deprecated in
favour of explicit Q/DQ models; they still work on TensorRT 10.3, JetPack 6.1/6.2's.)

### profile

On the device, per engine, for the first evaluation frame (a real input, so a model with data-dependent work such as
NMS times realistically):

```
trtexec --loadEngine=engines/NAME/engine.plan --iterations=1000 --warmUp=2000 --duration=10 --avgRuns=1
        --useSpinWait --loadInputs='INPUT':engines/NAME/input.bin --exportTimes=engines/NAME/times.json
        [--useDLACore=N] [--shapes=...]
```

- **Latency**: median, p95, p99, mean, standard deviation, min, max over every timed run in `times.json`, after the
  warm-up (2 s by default). `latencyMs` is trtexec's per-inference latency (host-to-device copy + GPU compute +
  device-to-host copy); `computeMs` (the GPU compute alone) is shown beside it. At least `--runs` runs and at least
  `--min-seconds` seconds.
- **Throughput**: trtexec's own `Throughput: … qps` line.
- **Power**: the Orin's **INA3221** monitors, read by the tool from sysfs. On L4T r36 that is the hwmon interface:
  `/sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*/inN_label` (the rail's name), `inN_input` (mV) and `currN_input` (mA),
  power = V × I; channels labelled `NC` are skipped. (L4T r32's iio interface, `in_powerN_input` in mW, is also read;
  without sysfs access `tegrastats --interval` is the fallback.) A thread samples every **20 ms**
  (`--power-interval-ms`); the chip's own update period (`update_interval`, ms) is recorded — sampling faster than it
  repeats values. The **window** is from trtexec's "Warmup completed" line to its "Timing trace has" line, as the tool
  reads them on the device (so loading and warm-up are excluded). An **idle baseline** (3 s, `--idle-s`) is sampled
  first.
- **Energy per inference** = mean power over the window (the **sum of the reported rails**, as jtop sums them) ×
  the median latency, in mJ — **ESTIMATED**, and labelled so everywhere: the rails sample the whole module at 20 ms,
  not one inference. The dynamic energy ((run − idle) × latency) and each rail's share are shown beside it. Whether the
  reported rails together are the module's whole input is the board's documentation's question (on the AGX Orin
  developer kit they are `VDD_GPU_SOC`, `VDD_CPU_CV`, `VIN_SYS_5V0` and `VDDQ_VDD2_1V8AO`); the gate can name the
  rails it sums instead.
- **Memory**: `MemAvailable` before the run minus its lowest value during it (system-wide; on a Jetson the GPU shares
  this memory, so it includes the engine's), the tool's peak RSS, and trtexec's memory lines.
- **Under what**: the power mode (`nvpmodel -q`, else `/var/lib/nvpmodel/status`), each devfreq device's and CPU
  core's current, minimum and maximum frequency (`jetson_clocks` pins minimum to maximum: that is how *locked* is
  judged), the thermal zones' temperatures before and after, the load average.

### accuracy

1. **The reference**: the registered ONNX model on this machine's CPU (onnxruntime's CPU provider, whatever GPU this
   machine has), with `core.models`' preprocessing and decoder, on the evaluation frames →
   `reference/predictions.jsonl`.
2. **Each engine on the device** (`engine_device.py infer`): each stored frame normalised by
   `falcon_models.normalise()`, run, decoded by `falcon_models.decode_detections()` with the frame's transform back
   to image pixels → `engines/NAME/dets.jsonl` (boxes in Falcon's classes, image pixels), brought back.
3. **Scored here** by `core.harness` on exactly those frames against the drives' ground truth: COCO AP, AP50, AP75,
   AR100 with 95 % bootstrap intervals over whole drives (`accuracy/NAME/predictions.jsonl`).
4. **Against the reference**: a **paired** bootstrap over drives (both resampled together) of each engine's AP minus
   the reference's (`core.harness.compare`) and of the **ratio** AP / reference AP (the gate's number), and how many
   frames have detections identical to the reference's.

Because the decoder is the same code, an engine whose outputs equal the reference's produces identical boxes: the
simulated fp32 engine agrees on every frame (tests). `core.models.nms` sorts equal scores stably, so ties come out in
the same order under JetPack's numpy 1.21 as under this machine's numpy 2 — found by running the device code in
Python 3.10 / numpy 1.21 (the `ros:humble-ros-base` container), where the unstable default sort ordered ties
differently.

### gate

A gate file (`falcon deploy gate --write-default gate.json` writes this one):

```json
{
  "schema": "falcon.deploy-gate/1",
  "name": "release gate",
  "accuracy": {"metric": "AP", "min_relative": 0.98, "rule": "ci_lower", "min_drives": 5, "min_reference": 0.05},
  "latency": {"max_p95_ms": 33.3, "measure": "latency"},
  "energy": {"max_mj_per_frame": 1000.0, "rails": "sum"},
  "power_mode": null,
  "require_locked_clocks": false,
  "missing": "incomplete"
}
```

| field | meaning |
|---|---|
| `accuracy.metric` | AP, AP50 or AP75 |
| `accuracy.min_relative` | the engine's metric must be at least this fraction of the reference's |
| `accuracy.rule` | `ci_lower`: the **lower end of the paired 95 % interval** of the ratio must reach it (non-inferiority shown); `point`: the estimate must |
| `accuracy.min_drives` | fewer evaluation drives than this: the criterion is *not measured* (a drive-level interval over few drives is not trustworthy; one drive gives a frame-level interval, which is too narrow) |
| `accuracy.min_reference` | a reference metric below this: *not measured* (a ratio to a near-zero AP is noise) |
| `latency.max_p95_ms` | the p95 of `latency` (trtexec's latencyMs) or `compute` (computeMs). Default 33.3 ms: a 30 Hz camera's period |
| `energy.max_mj_per_frame` | the estimated energy per inference; `rails`: `"sum"` or a list of rail names. Default 1000 mJ: a 30 W budget at 30 Hz — set your own |
| `power_mode` | optional: the recorded nvpmodel mode name must equal it (e.g. `"MAXN"`) |
| `require_locked_clocks` | optional: the clocks must have been locked (jetson_clocks) |
| `missing` | a criterion that could not be measured makes the verdict `incomplete` (default) or `fail` |

Each criterion is **pass**, **fail**, **not measured** (with why: an error, no power rails, too few drives, a
too-small reference) or **not set**. The verdict per engine: **fail** if any criterion fails, else **incomplete** if
any could not be measured, else **pass** — with the evidence (value, limit, interval, units, the rule) kept in
`deploy.json` and shown in the report. On the simulated device the verdicts are marked *not a release decision*.

### report

`falcon deploy report DIR` (also `falcon report deploy DIR --out FILE`, and the Reports page's *Deployment
(TensorRT)*): the device and its software, the power mode and clocks, the power source, the frames; per engine its
size, layers by precision, latency, throughput, power and estimated energy; the DLA fallbacks; the accuracy table with
intervals and an interval chart against the reference; the gate with its file and every criterion's evidence;
provenance and the SHA-256 of every file read.

## Files

```
deploy/<time>-<model>/                 (Models ▸ Deploy writes here; --out anywhere)
  deploy.json            model (path, SHA-256), device (mode, "from the user's settings"), env, plan, notes,
                         calibration, engines {name: build, profile, accuracy}, reference, gate, stages, provenance
  gate.json              the gate evaluated last
  report.html
  bundle/                what went to the device (eval/ and calib/ frames, spec.json, the tool, falcon_models.py)
  engines/<name>/        build.json build.log layers.json profile.json profile.log dets.jsonl (from the device)
  reference/predictions.jsonl, accuracy/<name>/predictions.jsonl
  simdevice/             the simulated device's folder (only with --device sim)
```

## Commands

```sh
falcon deploy check [--device sim|orin]
falcon deploy build PROJECT --model ID --run RUN... [--calib-run RUN...] --out DIR [--device sim|orin]
                    [--precision fp32,fp16,int8] [--dla 0|1] [--calib-count 500] [--every N] [--camera C]
                    [--workspace-mib 4096] [--threads N] [--seed S]
falcon deploy profile DIR [--runs 1000] [--warmup-ms 2000] [--min-seconds 10] [--idle-s 3] [--power-interval-ms 20]
falcon deploy accuracy DIR [--boot 1000] [--seed 0]
falcon deploy gate DIR [--gate FILE]          |   falcon deploy gate --write-default FILE
falcon deploy report DIR [--out FILE]
falcon deploy all PROJECT ... (build's options, profile's, --boot, --gate)
```

Protocol lines on stdout (docs/cli.md); exit 0, or 2 with `failed: …`.

On the page: Models ▸ **Deploy** — the model (registered ONNX detectors), the device, the precisions, a DLA core, the
calibration frames, every n-th frame, the timed runs, the gate file, the evaluation and the calibration drives; *Build,
profile, score and gate* runs `falcon deploy all` as a job (no GPU lock: the engines run on the device, or on the CPU
when simulated). Per deployment: the device and its software, the power mode and clocks, and per engine its latency,
throughput, power, estimated energy, AP with its interval, the ratio to the reference with its paired interval and
the gate's verdict; selecting an engine shows its build, layer precisions, DLA fallbacks and each criterion.

## What you must do for the real run

1. **The device and SSH** — already handled by the Hardware page: `falcon hil device --address ORIN --ssh-user USER
   --ssh-key ~/.ssh/id_ed25519` after `ssh-copy-id` (docs/hardware.md). Nothing else is stored.
2. **On the Orin, once**: JetPack 6.1 or 6.2 with TensorRT 10 — `sudo apt install nvidia-jetpack` installs it,
   including `python3-libnvinfer` (the Python API: needed for INT8 calibration and for fast accuracy runs) and
   `trtexec` at **`/usr/src/tensorrt/bin/trtexec`** (Falcon looks there, on the PATH, in `/usr/local/bin` and
   `/usr/bin`). `python3-numpy` (usually present). Check with `falcon deploy check --device orin`: it prints what the
   device has, including `trtexec`, `tensorrt_python`, the power rails and the power mode. (JetPack 6.0 ships TensorRT
   8.6: the tool handles its explicit-batch flag, but only TensorRT 10 is the target.)
3. **Lock the power mode and clocks before measuring, and keep them for every run you compare**:
   `sudo nvpmodel -m 0` (MAXN on the AGX Orin; `nvpmodel -q` shows the mode) and `sudo jetson_clocks`. Falcon records
   the mode and whether the clocks were locked with every profile, and the gate can require both (`power_mode`,
   `require_locked_clocks`). Let the device cool between long runs if the temperatures before and after differ much.
4. **Drives**: five or more evaluation drives (the gate's `min_drives`), and separate calibration drives with about
   500 frames between them.
5. **Run**: `falcon deploy all PROJECT --model ID --device orin --run … --calib-run … --out DIR [--dla 0]
   [--gate gate.json]`, or the page with *Jetson AGX Orin* chosen. Builds can take minutes per engine on the Orin.

What that run will measure for the first time: whether the TensorRT backend runs as written (the calibrator, the
ctypes CUDA calls, the output allocator for data-dependent shapes), real build logs and layer precisions, DLA
fallbacks, latency and throughput on the Orin's GPU and DLAs, power on its rails and the estimated energy, and the
accuracy of real TensorRT FP16/INT8 engines against the host's ONNX model.

## Measured here (simulated device: not TensorRT, not the Orin)

All on this machine's CPU with onnxruntime 1.28, 2026-10-29; AMD Ryzen 9 9900X, shared with other jobs.

**Synthetic drives (the tests)**: five drives of six 128 × 96 frames with grey squares as cars, and a real 4-node ONNX
detector built in the test (grey conversion, a 16 × 16 average pool); 30 evaluation frames, 10 calibration frames from
two other drives; 300 bootstrap resamples, seed 0. The reference scores AP 0.925 [0.898, 0.970]. fp32: identical
detections on 30 of 30 frames, ratio 1.000 [1.000, 1.000]. fp16 (weights rounded): AP 0.925, no change. **int8**
(entropy-calibrated QDQ): AP 0.865 [0.820, 0.903], AP − reference −0.060 [−0.096, −0.031], ratio 0.935 [0.898, 0.967] —
onnxruntime quantises the box coordinates on the model's output to 8 bits, which costs IoU at COCO's strict thresholds;
the gate fails it on accuracy (the interval lies below 0.98), and every engine is *incomplete* on energy (no rails
here).

**Real Town10HD recordings** (the M5 drives: an 8 s drive and a lead-brake scenario run, 1280 × 720) with a real
detector graph: the training lab's fasterrcnn_mobilenet_v3_small_fpn at 160 × 96 (BSD-3-Clause; trained on 40
fake-CARLA frames, so it **does not detect Town10's objects**: reference AP 0.0001). 50 evaluation frames (every 8th),
64 calibration frames (from the same two drives, evaluation frames excluded — noted as possibly optimistic), 300+
timed runs after a 1 s warm-up, 4 onnxruntime threads, load average 4–7:

| engine (simulated) | size | ONNX initialisers | median / p95 / p99 (ms) | throughput | AP [95 %] | AP / reference [95 %] | identical frames | gate |
|---|---|---|---|---|---|---|---|---|
| fp32 | 14.3 MiB | 126 float | 20.47 / 24.48 / 29.25 | 47.3 qps | 0.0001 [0.0000, 0.0004] | 1.000 [1.000, 1.000] | 50 of 50 | incomplete |
| fp16 (weights rounded) | 14.3 MiB | 126 float | 20.78 / 24.25 / 29.76 | 47.0 qps | 0.0001 [0.0000, 0.0004] | 0.717 [0.717, 0.896] | 0 of 50 | incomplete |
| int8 (QDQ, entropy) | 4.1 MiB | 369 int8, 375 float, 138 int32 | 19.38 / 22.57 / 25.15 | 50.2 qps | 0.0000 [0.0000, 0.0000] | 0.013 [0.006, 0.013] | 0 of 50 | incomplete |

300 timed runs each; power and energy *not measured* (no rails); latency passes the default 33.3 ms (on this CPU);
calibration on 64 frames took 8.4 s, the accuracy stage 124 s (three bootstraps of 1000 resamples).

The accuracy criterion is *not measured* for every engine — the reference AP is below the gate's `min_reference`
and there are only two drives — which is the honest answer for this model; the run shows the pipeline on real frames
and a real two-stage detector (dynamic shapes, NMS and RoIAlign inside the graph), including INT8 quantisation of it.
These latencies are a shared desktop CPU's and say nothing about the Orin.

## Limits

- **Detectors only** (2D boxes, registered as ONNX, output `boxes_scores_labels` or `yolo`); no segmenters, no
  TorchScript or plug-ins (TensorRT takes ONNX).
- **Batch 1**, one input tensor, a fixed input size (dynamic dimensions are fixed to the registration's size with
  `--shapes`).
- The TensorRT backend has **not run yet** (no Orin, no TensorRT here); the parsers of trtexec's layer information and
  logs are tested on hand-written samples of their documented formats, and the power sampler on a hand-made sysfs
  tree — real output may differ in detail, and the raw files are always kept.
- Energy is an **estimate** (mean module power × median latency), from 20 ms samples of rails that cover the whole
  module; it is not a per-inference measurement.
- INT8 uses TensorRT's implicit quantisation with a calibrator (deprecated in TensorRT 10 in favour of explicit Q/DQ
  models, still supported); quantisation-aware models with their own Q/DQ nodes build as they are.
- The simulated device's fp16 and int8 are stand-ins (see above), and its latency is this machine's CPU's.
- One camera per deployment, as the harness.
