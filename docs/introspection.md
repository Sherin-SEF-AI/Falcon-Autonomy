# Introspection: what a driving model attends to, and what it relies on

A driving model can score well and still be right for the wrong reason. This page is about looking inside: which
tokens a transformer policy attends to, where those tokens are in the world, and — because attention is not
explanation (Jain & Wallace 2019; Wiegreffe & Pinter 2019) — what happens to its steering and acceleration when one
token, one head or one input is changed. The closed-loop audits that turn such changes into a verdict are
[bias audits](bias-audit.md).

Code: `core/tokenpolicy.py` (the token transformer policy and its behaviour cloning), `core/tokenfeatures.py` (the
token observations both Pythons share), `core/introspect.py` (hooks, rollout, 3D projection, interventions, frames
files, the CLI), `ui/introspect_tab.py` (Models ▸ Introspection), `ui/view3d.py` (the 3D overlay). Each module's
docstring is the reference; this page is the guide.

```sh
falcon policy bc --arch token PROJECT --out DIR [--episodes N] [--scenario K]... [--dagger R] [--no-image]
                 [--net KEY=VALUE]... [--option KEY=VALUE]...
falcon introspect attention --policy DIR (--frames FRAMES.npz | PROJECT --scenario K [--seed S] [--steps N])
                            [--every N] --out DIR
falcon introspect attention --vlm hf:FOLDER --image IMG [--prompt TEXT] [--depth DEPTH.npy] --out DIR
falcon introspect intervene --policy DIR --frames FRAMES.npz --op JSON [--op JSON]... [--counterfactual F | --shift N]
                            --out DIR
falcon introspect audit PROJECT --policy DIR|follower|none --out DIR [--arms …] [--seeds N]    (bias-audit.md)
```

## The token policy

Today's end-to-end network (`core/e2e.py`) is a CNN: there is nothing to attend with. The token policy is a small
transformer whose inputs are separate, named tokens:

| Tokens | From | Each |
|---|---|---|
| image patches | the camera (128 × 72 by default) through a ResNet-18 trunk cut after layer 3 (stride 16): 8 × 5 = 40 cells | a 256-channel cell projected to d_model, with learned row and column embeddings |
| objects (≤ 16) | the environment's object tokens (`core/tokenfeatures.py`): the nearest road users within 60 m | class (car, truck, bus, two-wheeler, bicycle, autorickshaw, pedestrian, other), position and heading in the ego's frame, velocity, box, **brake-light bit**; empty rows masked |
| route | the state vector's route features | lateral offset, heading error, lane width, curvature at 10/25/50 m, remaining route, the light ahead |
| ego | the state vector's ego features and the **speed history** (the last 10 steps) | speed, target speed, acceleration, history |

Each token gets a type embedding; four pre-norm transformer blocks of eight-head self-attention (d_model 128 by
default) encode them; a learned plan query cross-attends (eight heads) to the result and two tanh heads give steer and
accel. The policy deliberately does **not** get the state vector's lead and vulnerable-road-user summaries (`lead_*`,
`vru_*`): it must find the road users in its object tokens or its image — so what it relies on can be measured.

The trunk starts from random weights and trains end to end. torchvision's ImageNet weights are used only when the net
spec names a `core.training.WEIGHTS` entry already on disk with its pinned SHA-256 — ResNet-18 is not pinned in this
build, so nothing is ever downloaded here (a pin is a patch request to `core/training.py`'s owner).

**Behaviour cloning in the environment.** The observations come from the learning environment itself
(`FalconDriveEnv` with `tokens` on), not from recorded drives: a privileged expert (`StateExpert` — lane keeping on
the route, IDM on the lead's gap and closing speed, stops for a red light and a vulnerable road user, reading the
state features the student does not get) drives episodes of lead_brake, cut_in, autorickshaw_cut_in and free driving;
each step is a sample. DAgger rounds (Ross, Gordon & Bagnell 2011) let the student drive with probability 1 − β
(β halving per round) while the expert labels. MSE, AdamW, validation split by episode, the best epoch kept; the
policy folder is an ordinary one (`net.kind: "token"`, `tokens`, `camera`), so `falcon policy rollout`, the
Leaderboard, the planner benchmark and the adversaries' "agent" ego run it, opening the environment with its tokens
and camera.

In the expert's data, a lead's brake light comes on exactly when it decelerates (brake lights follow deceleration in
the environment, `brake_lights: auto`). A cloned student may therefore learn to brake for the light — the causal
confusion of de Haan, Jayaraman & Levine 2019 — which is what the brake-light audit measures.

## The environment's token observations

`gym_config(..., tokens=True | {max_objects, range_m, history})` adds three observation arrays (workers/gym.py):
`objects` (16 × 18 float32), `object_ids` (the CARLA actor ids, −1 for empty rows) and `ego_history` (the ego's
speed at the last 10 steps). `camera: {…, depth: true}` adds an aligned planar depth image (CARLA's depth camera at
the same pose and lens, metres) — what puts attention in 3D. A vehicle's brake-light bit is the Brake flag of its
`VehicleLightState`: the director sets it for scenario vehicles as they decelerate (and as a bias audit's
perturbation says), level-k traffic as it brakes, and CARLA's traffic manager only when it manages the lights
(`vehicle_lights: tm`; otherwise CARLA leaves them off).

## Hooks

Every attention module gets a forward hook that keeps its weights (batch, heads, queries, keys):

* the token policy: the four encoder layers and the plan query (`core.tokenpolicy.Attention` returns its weights);
* a Hugging Face VLM (Qwen2-VL) loaded with `attn_implementation='eager'` (`HFRunner(..., attn_implementation=…)`,
  or the runner spec `hf:FOLDER,attn_implementation=eager`; the CLI adds it): only eager attention materialises the
  weights — SDPA and flash attention never form them. Its hook points are the attention modules that hold a LoRA
  adapter's wrapped projections (`core.vlm_lora.lora_modules`, their parents) when an adapter is loaded, else every
  attention module with `q_proj` and `o_proj` outside the vision tower.

A head is zeroed by a forward pre-hook on the attention's output projection that zeroes the head's slice of its
input — the same mechanism on both models. Hooks are removed when a probe returns.

**Attention rollout** (Abnar & Zuidema 2020): per layer the head-averaged weights with the residual added, Â =
½ (A + I), rows renormalised, multiplied through the layers; a token's relevance is the plan query's head-averaged
attention times the rollout (normalised to 1 over the tokens). For a VLM: the rollout row of the last prompt
position over the image tokens (Qwen2-VL merges 2 × 2 patches; the heat map is the merged grid).

## In the world

A patch cell's centre (u, v) is unprojected through `core/lens.py` (the camera's intrinsics), scaled to the median
planar depth of the cell, moved by the camera's pose (`lens.optical_to_vehicle`) into the vehicle frame and by the
ego's pose into the world. Cells beyond 80 m (the sky) are dropped. The object tokens are their road users' boxes.
`attention.json` holds both frames: `points` and `objects` in the vehicle frame (what Models ▸ Introspection draws on
`ui/view3d.py`, coloured by relevance on the theme's sequential scale) and `points_world`, `objects_world` with the
frame's `ego` pose.

## Interventions

`falcon introspect intervene` applies ops to every frame of a recording (a frames file: every observation array of
an episode stacked, the ego's poses, the actions, the seed and options — so a probe is reproducible offline) and
reports per frame the action without and with them, and the mean and largest |Δsteer| and |Δaccel|:

| Op | What changes |
|---|---|
| `{"op": "mask_object", "id": N}` | one road user's token is hidden from every query (exactly as if its row were empty) |
| `{"op": "mask_class", "class": "autorickshaw"}` | every token of a class |
| `{"op": "mask_tokens", "type": "patch" \| "object" \| "route" \| "ego", "index": [k, …]}` | tokens by type (all without index) |
| `{"op": "zero_head", "layer": L \| "plan", "head": h}` | one head's output |
| `{"op": "patch", "layer": L, "tokens": "object" \| "patch" \| "all" \| [k, …]}` with `--counterfactual F` or `--shift n` | activation patching (Vig et al. 2020; Meng et al. 2022): the residual stream after layer L at those tokens taken from the counterfactual observation |
| `{"op": "flip", "feature": "brake_light", "id": N \| "class": C \| "all": true}` | a binary object feature flipped |
| `{"op": "set", "feature": F, "id": N, "value": V}` | an object feature set |
| `{"op": "speed_history", "offset": m/s}` | the ego's speed history shifted, the scene unchanged |

The VLM's counterpart is `vlm_zero_head` (the KL divergence of the next-token distribution and its top token before
and after).

## Models ▸ Introspection

The tab (the Models page's optional tab, `ui/introspect_tab.py`) lists the workspace's token policies, trains one
(`falcon policy bc --arch token`), records an episode and its attention, browses the frames — the most relevant
tokens in a table, the patch points and object boxes in the 3D view — runs an intervention on the last recording
(mask the autorickshaws, turn the brake lights off, shift the speed history, hide the image, zero a head) and runs a
bias audit of any ego, its table of reliance ratios with intervals. Each action is a job; outputs land in the
workspace's `introspect/` folder (`attention-…`, `intervene-…`, `audit-…`). The assistant's `formal_causal_probe`
queues the same jobs and `formal_causal_result` reads them back.

## Measured

On the fake carla (the test suite): a random-initialised token policy (d_model 64, 8 heads, 4 layers) — hook shapes
(1, 8, 58, 58) per layer and (1, 8, 1, 58) for the plan query on 40 patches + 16 object slots + route + ego; empty
object rows get exactly zero attention; masking and head zeroing are deterministic, masking a token equals removing
its row (to 10⁻⁶), patching all tokens after the last layer gives the counterfactual's action (to 10⁻⁵); patch rays
at a planar depth of 10 m land 11.5 m ahead of the vehicle origin with the expected lateral offsets. A tiny random
Qwen2-VL (two 4-head language layers, no download) with eager attention: both layers hooked, 6 merged image tokens
over a 2 × 3 heat map, deterministic, a zeroed head changes the next-token distribution; with a LoRA adapter the hook
points are its attention modules and, the adapter starting at B = 0, the relevance is unchanged. Training on the real
server and real VLM weights are acceptance steps.

## Limits

A patch's position is its cell's centre (a ResNet cell sees more than its cell); rollout assumes a residual share of
½; the VLM path reads the language model's attention to the image tokens from the last prompt position; patching
replaces whole token vectors at one layer; the expert is a rule set; images from the fake carla are flat renderings.
