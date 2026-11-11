# Performance profiles

A **profile** is a named set of settings values: **standard** (the settings as you set them) or **low_vram**, for a GPU
with about 8 GB of video memory. Settings ▸ **Performance**; `falcon profile show|set|measure`. Code:
`falcon_autonomy/core/profiles.py`, the FP16 runners and the gate in `core/models.py`, the VRAM sampler in
`core/sysinfo.py`.

## What low_vram sets

| Setting | low_vram | standard default | Where it acts |
|---|---|---|---|
| `server_quality` | Low | Epic | the server's `-quality-level=Low` |
| `texture_pool_mb` | 512 | 0 (CARLA's own) | Unreal's texture streaming pool, from the next server start |
| `render_budget_mpx` | 96 | 200 | runs rendering more megapixels per step are refused before they start |
| `server_memory_gb` | 10 | 12 | the server's memory cap (its systemd scope; RAM, not video memory) |
| `inference_precision` | auto | fp32 | model evaluations (`falcon evaluate`) |

and caps for 3D reconstruction (`RECON_CAPS`: at most 300 000 Gaussians, a 480 px wide training image, 150 000 initial
points, half the GPU), applied to `falcon recon train`'s settings (`profiles.cap_recon()`).

`falcon profile set low_vram` (or choosing it and pressing Apply) writes these into the settings and keeps the values
it replaced in `profile-saved.json` beside `settings.json`; `set standard` puts them back. Every part of Falcon reads
the settings as it always did, so a value you change afterwards stays changed: `falcon profile show` lists it as a
difference from the profile.

**The texture pool** is passed both ways Unreal accepts it,
`-ini:Engine:[/Script/Engine.RendererSettings]:r.Streaming.PoolSize=512` and `-ExecCmds=r.Streaming.PoolSize 512`
(the same value). Which of the two CARLA 0.9.16's packaged server honours has not been measured. 512 MB is half UE4's
default pool (1000 MB), chosen, not measured.

## FP16 inference

`Runner(model, precision='fp16')` runs an ONNX model converted by onnxruntime's
`float16.convert_float_to_float16(keep_io_types=True)` (inputs and outputs stay FP32), cached once per weights file as
`<SHA-256 of the FP32 file>.fp16.onnx` under `$XDG_CACHE_HOME/FalconAutonomy/fp16/`; a TorchScript model is `.half()`
(under `torch.autocast` on CUDA). A model that does not load in FP16 runs in FP32 and says why; a plug-in runs at its
own precision.

**auto** keeps FP16 only when it holds: the model runs on the evaluated frames in both precisions, each is scored
against the ground truth (COCO AP; a segmenter's mIoU), and FP16 is kept when it is at most **0.5 points** below FP32,
else the evaluation is FP32's. The two runs are the evaluation itself — nothing runs a third time. `evaluation.json`
records `precision` (what it ran at, what was asked, both scores, the drop and the reason) and its provenance settings
the precision used.

On the fake (one frame, a detector whose boxes are that frame's cars): FP32 AP 100.0, FP16 AP 100.0, drop 0.00 points
→ FP16 kept; the same detector with a score computed from constants that FP16 conversion clamps (to ±10 000): FP16 AP
0.0, drop 100.00 points → FP32.

## `falcon profile`

```
falcon profile show [--json]          the profile in force, its values and what differs from it
falcon profile set standard|low_vram  switch (prints what changed)
falcon profile measure [--seconds S] [--interval S] [--limit-gb G] [--json] [-- COMMAND …]
                                      the GPU's peak used video memory while COMMAND runs (or for S seconds), sampled
                                      with nvidia-smi; 2 when the peak is above G (default 7.5)
```

`falcon doctor` suggests low_vram on a GPU with 9 GB or less.

## Limits

- No number here was measured on an 8 GB card. An 8 GB limit cannot be enforced on the 16 GB reference card, so the
  acceptance target is a **measured peak of at most 7.5 GB** over an 8-step loop on it (`falcon profile measure --
  falcon loop …`); that measurement is still to be made on the real server.
- `measure` samples the whole GPU's used memory (other programs' too) every interval: a peak shorter than the interval
  can be missed.
- The FP16 gate is as good as the evaluated frames: few frames or few objects make its AP comparison coarse.
