# World-model and end-to-end training

Three ways to learn driving from Falcon's derived data, each judged honestly against something simpler:

- **A learned world model** (`core/wmtrain.py`, networks in `core/wmnet.py`, data in `core/wmdata.py`). It predicts
  future BEV frames, rewards and episode ends from the actions. It is judged on held-out drives against copying the
  last frame and against constant-velocity extrapolation.
- **Policy learning in imagination.** Dreamer's actor-critic runs inside the learned model. The resulting policy is
  judged in the real environment against behaviour cloning and PPO given the same data budget.
- **End-to-end driving** (`core/e2e.py`). A camera (+ BEV) to control policy is learnt by behaviour cloning from
  recorded drives. It is judged offline (action error by drive) and online (Leaderboard-style scores).

The full definitions are in the module docstrings. The measured numbers are below, in *Measured*.

```sh
falcon worldmodel collect PROJECT --out DIR [--behaviour scripted|random|POLICY] [--episodes N] [--scenario K]
                                   [--start-speed KMH] [--sigma S]          # environment episodes (needs CARLA)
falcon worldmodel train (--out DIR | --workspace W [--id ID]) --run SRC... [--val SRC...] [--iterations N]
                        [--seed S] [--device auto|cpu|cuda] [--option KEY=VALUE]   # SRC: a drive or experience folder
falcon worldmodel eval --model DIR [--run SRC...] [--horizon 20] [--stride 5]      # default: the held-out units
falcon worldmodel imagine --model DIR --out POLICY [--iterations N] [--horizon H]  # a policy folder
falcon worldmodel e2e --run RUN... --out POLICY [--epochs N] [--bev] [--device D]  # end-to-end cloning
falcon worldmodel e2e-offline --policy POLICY --run RUN... [--out DIR]
falcon worldmodel show DIR                                                         # a model's or evaluation's table
falcon policy rollout | leaderboard PROJECT --policy POLICY ...                    # any of these policies, online
```

Training, evaluation, imagination and end-to-end cloning are `needs_gpu` jobs. From the command line they take the GPU
lock and the CARLA lock through `core.training.gpu_locks`, as detector training does. So CARLA jobs queue behind
them, and they wait for a running CARLA job. The policies are ordinary policy folders (`core/policy.py`), so the
Leaderboard, rollouts and the infraction replay run them unchanged.

## Data

A **sequence** holds one row per step, in DreamerV3's convention:

- the observation: the BEV raster's 8 planes and the environment's state vector;
- the action that led to it;
- the reward received on arriving at it;
- whether it is terminal.

The BEV is `core.worldmodel`'s own raster, the one `falcon derive worldmodel` and the environment use. It is drawn at
the environment's spec by default: 0.5 m pixels, 32 m ahead, 8 m behind and 16 m to each side (80 × 64). The state
vector is `core.drivefeatures`' observation, built by replaying the drive through the environment's featuriser, as
behaviour cloning does.

**From a recorded drive:**

- **Actions.** The recorded control (drives since 0.6). A drive without control gets **derived** actions, labelled
  `action_source: derived`. They come from inverting the environment's kinematic bicycle model on the recorded speeds
  and yaws: 3 m/s² full throttle, 8 m/s² full brake, steer × 70°, wheelbase 1.2 × the box's half length. This is exact
  for a kinematic drive. For a physics drive (the autopilot's) it is an approximation.
- **Reward.** `drivefeatures.reward_terms` with the environment's default weights, from the replayed facts:
  - progress, speed, lane offset, heading, jerk and steer rate;
  - collisions by box overlap;
  - off road: the ego's centre outside lanes.json's drivable raster (the environment asks CARLA's map);
  - route deviation and agent blocked.

  `wrong_way`, `outside_lanes`, `red_light` and `stop_sign` cannot be computed from a recording. They are 0, and the
  report lists them.
- **Terminal.** Never. A recording ends by time, so the continue head learns episode ends only from environment
  episodes.

**From the environment** (`falcon worldmodel collect`). A behaviour drives the learning environment with the BEV on.
Each step keeps the observation, the applied action, the environment's own reward and terms, and `terminated`. A time
limit is not terminal. The behaviours:

- `scripted`: a proportional lane-keeping and speed controller on the state vector, with heuristic gains;
- a policy folder;
- `random`.

Each has Ornstein–Uhlenbeck exploration noise added (θ 0.15, σ 0.2 by default). These episodes carry what recorded
drives do not: off-policy actions, crashes, leaving the road, and the environment's exact reward.

## The world model

A recurrent state-space model in the form of **DreamerV3** (Hafner, Pasukonis, Ba and Lillicrap 2023, "Mastering
Diverse Domains through World Models", arXiv:2301.04104). The RSSM itself comes from PlaNet (Hafner et al. 2019).

- **The latent.** A GRU deterministic state of 256 units and 16 categorical latents of 16 classes. The latents use
  straight-through gradients and 1 % uniform mixing.
- **The encoder.** Four stride-2 convolutions over the BEV, beside an MLP over the normalised state vector.
- **The heads.** A transposed-convolution decoder to 8 Bernoulli planes, a state head, a symlog reward head and a
  continue head.
- **The loss.** BEV binary cross-entropy (summed over pixels and planes), state squared error, symlog reward squared
  error, continue cross-entropy, and the KL split into a dynamics term (β 0.5) and a representation term (β 0.1),
  each with 1 free nat.
- **Optimisation.** Adam at 3·10⁻⁴, gradient clipping at 100, batches of 16 windows of 32 steps.

**Departures from DreamerV3**, chosen for this data:

- the BEV is binary, so its likelihood is Bernoulli rather than a symlog squared error;
- the reward is a symlog squared error rather than two-hot regression;
- the GRU is torch's own rather than layer-normalised;
- the convolutions use GroupNorm(1, C).

None of these settings was tuned.

**The split is by unit**, a whole drive or a whole experience folder, given as `--val` or drawn at random.
The **final weights are the last iteration's**. The held-out units choose nothing, and their loss is logged only as a
curve, so the evaluation on them is a held-out evaluation.

**Seeding and resuming.** A run is seeded: each iteration's windows and torch's random stream are functions of
(seed, iteration). On the CPU, two runs, and a run cancelled and resumed, give identical weights (tested). On CUDA they
are not bit for bit, because of cuDNN's kernels.

A model folder holds:

- `model.json`: the network, the features and their normaliser, the BEV spec, the units of each split, the cost and
  the provenance;
- `weights.pt`, its SHA-256 checked on loading;
- `metrics.jsonl`: the training curve;
- `data.json`: how every sequence was derived;
- `run.json`;
- `eval/`.

## Evaluation

**Open loop**, on the held-out units by default:

1. Each sequence is filtered through the posterior from its first step. The latents' mode is used, so the evaluation
   is deterministic.
2. From every start t0 = 5, 10, 15, … the model imagines forward with the sequence's own recorded actions for k = 1 …
   20 steps (1 s at 20 Hz), seeing nothing after t0.
3. A plane is predicted where its probability exceeds 0.5.

The measures:

- **BEV IoU per channel** and for "actors" (the union of the vehicle, pedestrian, cyclist and other planes). It is
  the pooled intersection over the pooled union at each horizon. A channel empty in both truth and prediction
  everywhere is "—".
- **Reward** mean absolute error.
- **Continue**: on environment episodes only, the Brier score of the continue probability.

The baselines, on exactly the same (start, horizon) pairs:

- **Copy the last frame** (and the last reward).
- **Constant velocity**, on recorded drives (it needs actors.jsonl):
  - every actor within the BEV window at t0, and the ego, moves at its recorded velocity with its yaw kept, and is
    redrawn in the extrapolated ego frame;
  - the map planes are the frame at t0, shifted by the ego's motion (a pixel from outside the window is empty).

  So it uses the frame at t0 and the velocities, never the map beyond the window. On a synthetic drive at exactly
  constant velocity, its actor planes equal the truth at every horizon (tested).
- **The training mean reward.**

`eval.json` has every curve, the pair counts, per-sequence numbers and a **verdict** per horizon and channel: whether
the model beats every baseline. `samples.npz` feeds the tab's viewer.

## Policy learning in imagination

Dreamer's actor-critic in the latent space:

1. **Start states** are posterior states of training windows.
2. **Rollout.** The actor's actions drive the frozen model's prior for 15 steps. Rewards come from the reward head and
   continues from the continue head.
3. **Returns.** λ-returns (γ 0.99, λ 0.95), bootstrapped with a critic.
4. **The actor's gradient** backpropagates through the model's dynamics (Dreamer, Hafner et al. 2020, "Dream to
   Control"). The return is normalised by DreamerV3's percentile range, and there is an entropy bonus of 3·10⁻⁴.
5. **The critic** regresses symlog returns, regularised towards a slow copy of itself.

**One deliberate difference: the actor reads the decoded state vector**, not the latent. So the trained actor is an
ordinary feed-forward policy over named features (`trained_by: dreamer`, net kind `mlp`), and the environment runs it
without the world model or a recurrent state. The price: in the real environment, a decoding error in imagination
becomes an input error.

## End-to-end driving

`falcon worldmodel e2e` clones behaviour from the camera to the recorded control.

**The pairs** are behaviour cloning's (each step's next control), each with:

- the camera image at that step, as RGB, area-resized to 128 × 72 (the environment's default camera);
- two state features: the speed (the measurement) and the target speed (the command), as conditional imitation
  learning gives them (Codevilla et al. 2018);
- with `--bev`, the BEV too.

`prev_*` are left out on purpose: they are the copycat shortcut.

**The network** is PilotNet's convolutions (Bojarski et al. 2016) with an MLP head. Training is seeded, the split is
by drive, and the epoch with the lowest validation loss is kept.

**The policy folder** has net kind `e2e` and a `camera` block: the drive's camera pose and field of view. `falcon
policy rollout`, `falcon policy leaderboard` and the Training page open the environment with that camera. Drives
recorded with different camera poses are refused together, since the environment can open one camera.

## Training ▸ World model

The left side has four sections:

- **Train** on checked drives and experience folders.
- **Collect environment episodes.**
- **Learn a policy in imagination** from the chosen model.
- **Clone an end-to-end policy** from the checked drives.

The right side shows the chosen model:

- its method, data, held-out units, cost and verdict;
- *Evaluate on the held-out drives*;
- a chart of one channel's IoU against the horizon for the model and each baseline (colour and marker shape);
- the BEV viewer: the truth, the model's prediction and a baseline side by side, over a horizon slider.

Policies made here appear in the Driving policies and Leaderboard lists.

## Measured

All on the reference machine (RTX 5080, 24 cores), 2026-10-29, seed 0. **No real CARLA server was used.** The real
data is the two Town10HD drives recorded in milestone 5; everything else is the fake carla, whose straight road and
flat renderings are not CARLA. The numbers show what the code does and where each method stands relative to the
others; they are not CARLA results.

### The data

- **78 scenario drives on the fake**: 13 built-in scenarios × seeds 1–6, the lane follower in kinematic motion, a
  128 × 72 camera at the environment camera's pose, and recorded control. Seeds 1–4 (52 drives, 13,760 steps) are
  for training, seeds 5–6 (26 drives) are held out.
- **Environment episodes with the scripted behaviour + noise**:
  - free driving from 30 km/h: 40 episodes, 11,702 steps, 15 off the road;
  - lead_brake: 20 episodes, 4,553 steps;
  - cut_in: 20 episodes, 1,347 steps, 9 collisions and 9 off the road. This folder is held out.
- **Derived actions checked against recorded control**, on all 78 kinematic drives (20,562 transitions): steer error
  0; accel error 4·10⁻⁵ while moving (largest 7·10⁻⁴), and 0.92 at the 1,176 stopped steps (a brake held at a
  standstill cannot be seen).

### The world model on the fake

**Training.** 6,000 iterations (batch 16 × 32 steps) on 30,015 training steps took 399 s on the GPU, peaking at
1.45 GiB. The model has 6.8 M parameters. The held-out set is 8,227 steps: 26 drives and the cut-in episodes. The
final training loss per window step was 361 (BEV 356 nats); the held-out loss was 804 (BEV 794). It generalises much
worse than it fits.

**Open loop, held out**: 1,604 starts at k = 1 and 1,436 at k = 20. The evaluation took 42 s on the CPU. IoU per
channel:

| channel | model k=1 / 5 / 10 / 20 | copy last frame | constant velocity |
|---|---|---|---|
| drivable | 0.989 / 0.973 / 0.951 / 0.911 | 0.982 / 0.935 / 0.915 / 0.923 | 0.991 / 0.959 / 0.919 / 0.838 |
| centerline | 0.765 / 0.589 / 0.499 / 0.444 | 0.806 / 0.767 / 0.772 / 0.788 | 0.991 / 0.959 / 0.919 / 0.838 |
| lane boundary | 0.849 / 0.596 / 0.545 / 0.512 | 0.808 / 0.770 / 0.777 / 0.791 | 0.991 / 0.959 / 0.919 / 0.838 |
| vehicle | 0.417 / 0.355 / 0.297 / 0.222 | 0.860 / 0.607 / 0.446 / 0.262 | 0.982 / 0.957 / 0.908 / 0.756 |
| pedestrian | 0 / 0 / 0 / 0 | 0.460 / 0.163 / 0.091 / 0.052 | 0.991 / 0.879 / 0.568 / 0.258 |
| cyclist | 0 / 0 / 0 / 0 | 0.669 / 0.220 / 0.077 / 0.042 | 0.994 / 0.922 / 0.755 / 0.491 |
| actors (union) | 0.401 / 0.343 / 0.288 / 0.216 | 0.852 / 0.591 / 0.431 / 0.254 | 0.982 / 0.956 / 0.900 / 0.739 |

| reward mean absolute error | k=1 | 5 | 10 | 20 |
|---|---|---|---|---|
| model | 0.145 | 0.124 | 0.132 | 0.110 |
| copy last reward | 0.164 | 0.135 | 0.141 | 0.123 |
| training mean | 0.205 | 0.182 | 0.185 | 0.162 |

Constant velocity scores the three map planes identically here. The fake's road is straight, so shifting the frame
empties the same share of rows in each plane; on the real drives below they differ.

The continue head scored a Brier score of 0.004–0.025 per horizon on the 20 held-out cut-in episodes (18 terminal
steps).

**Verdict: the learned model does not beat the baselines on the BEV.** It beats both only on the drivable plane at
k = 5 and 10, and loses everywhere else. On actors it is far below constant velocity at every horizon, and below
copying the last frame up to k = 20. The cause is not only the dynamics: **its reconstruction, given the
observation itself, reaches a vehicle IoU of 0.43 and 0 for pedestrians and cyclists.** The rare, small planes never
pass 0.5, and a predicted actor blurs away within a few steps (the viewer shows a cutting-in car vanish). **It does
predict the reward better than both baselines** at every horizon, by 8–12 % over copying the last reward.

**A larger model does not fix it.** DreamerV3's own latent size (32 × 32 categoricals, a 512-unit GRU, depth 48;
20.6 M parameters) was trained on the same split: 6,000 iterations in 494 s, peaking at 2.16 GiB. It reached actors
IoU 0.369 / 0.336 / 0.302 / 0.241 and a reconstruction vehicle IoU of 0.40, with pedestrians and cyclists still 0.
It beats both baselines on the drivable plane at k = 5, 10 and 20, and nowhere else. Its reward error, 0.144 / 0.119 /
0.132 / 0.115, matches the smaller model's. This second variant is reported beside the first, not chosen over it:
choosing on the held-out drives would make them a tuning set.

### The world model on real Town10HD

The two milestone-5 drives, both recorded without control, so their actions are derived:

- `drive`: 8 s of the autopilot with physics, 40 vehicles and 20 walkers;
- `lead`: 12 s of lead_brake.

Each fold trains on one drive and holds out the other: real only (3,000 iterations), or real plus the fake training
set (6,000 iterations). The GPU was shared with another job during these runs, so their times (300–670 s) are not
comparable with the fake run's.

| held out, actors IoU at k = 1 / 5 / 10 / 20 | model | copy last frame | constant velocity |
|---|---|---|---|
| `lead`, trained on `drive` (160 steps) | 0.07 / 0.02 / 0.01 / 0.02 | 0.91 / 0.64 / 0.38 / 0.17 | 0.995 / 0.96 / 0.85 / 0.57 |
| `lead`, trained on `drive` + fake (30,175) | 0.00 / 0.03 / 0.05 / 0.06 | same | same |
| `drive`, trained on `lead` (240 steps) | 0.01 / 0.01 / 0.01 / 0.01 | 0.68 / 0.08 / 0.01 / 0.18 | 0.83 / 0.44 / 0.22 / 0.11 |
| `drive`, trained on `lead` + fake (30,255) | 0 / 0 / 0 / 0 | same | same |

| held out, drivable IoU at k = 1 / 20 | model | copy last frame | constant velocity |
|---|---|---|---|
| `lead`, real only / real + fake | 0.42 / 0.40; 0.43 / 0.40 | 0.99 / 0.83 | 0.99 / 0.82 |
| `drive`, real only / real + fake | 0.44 / 0.40; 0.85 / 0.72 | 0.99 / 0.93 | 0.98 / 0.67 |

The reward error tells the same story. Copying the last reward gives 0.003–0.011; the models give 0.012–0.18.

**On real data the learned model fails outright.** One or two hundred steps teach it almost nothing, and the fake's
straight road does not transfer to Town10's junctions and curves. Mixing them made the actor planes worse. The one
partial exception is the drivable plane of the `lead` + fake model on `drive`: 0.72 at k = 20, above constant
velocity (0.67) but below copying the last frame (0.93).

One thing this shows about the baselines themselves: **constant velocity is worse than copying the last frame on
`drive`**. The autopilot turns there, and a straight-line extrapolation of the ego swings the whole frame wrong.

### Policy learning in imagination, against behaviour cloning and PPO

All policies were scored on the fake by `falcon policy leaderboard` over the same 20 routes: lead_brake, cut_in,
pedestrian_crossing, stalled_vehicle and cyclist_crossing with 3 seeds each, plus 5 free-driving routes from a
standstill. Intervals are 95 % bootstrap intervals over routes.

**The budgets:**

- **Imagination**: the world model's 30,015 training steps (the 52 drives and 16,255 environment steps of the
  scripted behaviour), then 1,500 imagination iterations (83 s on the GPU).
- **PPO**: 28 updates × 1,024 = 28,672 environment steps (3 min 42 s on the CPU). It is still in the default
  curriculum's first stage.
- **BC and end to end**: the 52 drives' recorded control. They cannot use environment episodes.

| policy | driving score | route completion | infraction penalty | successes | mean speed | env return | collisions |
|---|---|---|---|---|---|---|---|
| imagination (Dreamer-style) | **0.569** [0.373, 0.763] | 0.586 | 0.82 | 10 / 20 | **61 km/h** | −20.7 | 9 |
| behaviour cloning (state) | 0.156 [0.125, 0.190] | 0.159 | 0.98 | 3 / 20 | 23 km/h | −21.4 | 1 |
| PPO, same budget | 0.216 [0.103, 0.353] | 0.232 | 0.90 | 3 / 20 | 35 km/h | −23.0 | 6 |
| end to end, camera | 0.487 [0.352, 0.619] | 0.491 | 0.98 | **14 / 20** | 24 km/h | **−1.3** | 1 |
| end to end, camera + BEV | 0.233 [0.147, 0.333] | 0.252 | 0.94 | 5 / 20 | 27 km/h | −18.6 | 4 |

**The imagination policy has the best driving score, and it does not drive well.**

- **It drives much faster than asked.** Per route its mean speed was 1.05–2.4 times the target (median about 1.7;
  targets 23–70 km/h, 30 on free routes). It averaged 61 km/h, peaked at 84, and collided on 9 of 20 routes.
- **Why.** The recorded drives never overspeed, so the reward head learnt almost nothing of the speed term's
  overspeed penalty. The actor found that hole in imagination: the model-exploitation failure Dreamer is known for.
- **What hides it.** The Leaderboard's driving score counts completion and collisions, not speed, so racing through
  routes scores well. Its environment return (−20.7) is no better than behaviour cloning's.

**A second Dreamer round** does what Dreamer's loop does: collect with the current actor, refit the model,
imagine again.

1. **Collect.** 40 episodes (4,795 steps) with the round-1 policy plus noise: free driving, lead_brake and cut_in. It
   collided in 16 of them.
2. **Refit.** The model was refitted from scratch on all 34,810 steps (414 s). Its held-out actors IoU (0.410 / 0.369
   / 0.326 / 0.242) and reward error (0.145 / 0.112 / 0.122 / 0.110) barely moved.
3. **Imagine.** The imagined return fell from 1.15 to 0.04: the model now knows that speeding into cars costs.

PPO was resumed to the same budget: 34 updates, 34,816 steps.

| policy (round 2) | driving score | completion | penalty | successes | speed ÷ target (median) | env return | collisions |
|---|---|---|---|---|---|---|---|
| imagination, round 2 | 0.594 [0.422, 0.771] | 0.623 | 0.90 | 9 / 20 | 1.44 (round 1: 1.69) | −13.6 (−20.7) | 5 (9) |
| PPO, 34,816 steps | 0.225 [0.115, 0.360] | 0.289 | 0.80 | 2 / 20 | 1.09 | −37.7 | 10 |

**The second round helped and did not finish the job.** Collisions fell from 9 to 5, overspeeding from 1.69 to 1.44
times the target, and the return rose from −20.7 to −13.6. The policy still speeds, collides on a quarter of the
routes, and its return is far below the camera policy's −1.3. **With equal environment steps, it is ahead of PPO**
(PPO is still in its curriculum's first stage). **Neither is a driver.**

### End-to-end driving

| | held-out validation drives | 26 drives never used |
|---|---|---|
| end to end, camera: action MAE (steer / accel) | 0.120 (0.0016 / 0.239) | 0.133 (0.0013 / 0.264) |
| end to end, camera + BEV | 0.148 (0.0066 / 0.289) | 0.144 (0.0052 / 0.282) |
| behaviour cloning, state | 0.121 (0.012 / 0.229) | **0.104** (0.012 / 0.196) |
| constant baseline (mean action) | 0.139 | 0.132 |

- **Offline, the camera policy is no better than the constant baseline** on drives it never saw (0.133 against
  0.132): its accel error equals the baseline's. The state policy beats it. The camera + BEV policy is worse than the
  constant baseline.
- **Online, the camera policy did best on success rate and return** (14 of 20 routes, return −1.3), driving at
  24 km/h with one collision. **Low offline error does not predict closed-loop driving, in either direction.**
- **Cost.** Training took 13 s (camera) and 34 s (camera + BEV) on the GPU for 30 epochs over 10,161 pairs. The
  camera + BEV policy's best validation epoch was 3: it overfits fast.


## Limits

**The data:**

- A recorded drive carries only its driver's behaviour, the lane follower or the autopilot. A model trained on drives
  alone has never seen off-policy actions. Environment episodes with exploration noise are what cover them.
- A recorded drive's reward lacks four terms (wrong way, outside lanes, red light, stop sign), and its off-road test
  uses lanes.json instead of CARLA's map. Its reward is therefore not quite the environment's reward.
- Derived actions are exact for kinematic drives while moving. They are approximate for physics drives, and
  unobservable at a standstill: a brake held while stopped derives as 0.
- **The real data is two drives**: 160 and 240 steps of Town10HD, recorded in 0.5 without control. That is far too
  little to learn a world model from, and the numbers above say so.

**The model:**

- It reconstructs small objects poorly. Pedestrians, cyclists and "other" never pass the 0.5 threshold, even with the
  observation (reconstruction IoU 0). Vehicles reach about 0.43. A larger latent did not change that (*Measured*).
- No occupancy input. The occupancy grids of `falcon derive worldmodel` are not used.
- Its settings were not tuned.

**The policies:**

- **A policy learnt in imagination exploits the model's errors.** Measured: it drove at a median 1.7 times the target
  speed, because the data barely showed overspeeding.
- The Leaderboard's driving score does not penalise speeding. A high score can hide a policy no one should drive;
  read the speed and the collisions beside it.
- **End-to-end cloning copies its teacher** and suffers covariate shift. The fake's images are flat renderings,
  so its numbers say the code works, not that a camera policy would drive in CARLA.

**Where it was measured:** everything here ran on the fake carla's straight road and on two recorded real drives. No
real server was used in this milestone, so nothing was evaluated online in CARLA.

**Not verified:**

- training on CUDA bit for bit (cuDNN), and the CUDA resume;
- the GUI's jobs end to end against a real server;
- an end-to-end policy driving with CARLA's rendering (its camera spec and the environment's camera path are those
  the Leaderboard replay already uses, but not run on a server here).
