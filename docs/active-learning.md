# Active learning

`falcon experiment active` (Training ▸ *Active learning*) asks: **does choosing which frames to train on by what the
model does not yet know reach the same AP from fewer frames than choosing them at random?** Frames are chosen in
rounds with a fixed budget from a pool of recorded drives, by the model trained so far, and every strategy is measured
against random choice on the same pool and the same held-out test set, over several seeds, with paired intervals.
Code: `falcon_autonomy/core/activelearn.py` (its docstring is the reference).

**In Falcon the labels are free.** Every frame comes labelled from the simulation, so "labelling" a frame here means
*adding it to training*. The point is data efficiency — the same AP from fewer frames: shorter trainings, smaller
datasets, fewer drives to record — not saving annotation cost. Results say so.

## Acquisition strategies

Each round the model of the previous round scores every frame of the pool not yet in training, in evaluation mode, and
the round's budget of B frames is chosen.

| strategy | per frame | notes |
|---|---|---|
| `random` | a seeded draw | the baseline; always run |
| `entropy` | Faster R-CNN: the normalised entropy H(q) / ln(K + 1) of each proposal's softmax over the K classes and the background. FCOS / RetinaNet (independent sigmoids): the binary entropy of the highest class score, / ln 2 | over the proposals (anchors) whose best foreground score is at least the detector's own score threshold (0.05); aggregated per frame by `--aggregate` max (default), mean or sum; 0 when none |
| `margin` | 1 − (p₁ − p₂) of each proposal's two highest probabilities (the background included for Faster R-CNN) | same proposals and aggregation |
| `det_mean`, `det_max` | the mean or maximum over the frame's final detections scoring at least 0.3 of (1 − score) — the detection's score is its class probability, the maximum over classes | 0.3 is a heuristic floor; a frame with no such detection scores 0 |
| `kcenter` | diversity without a learned loss: k-centre greedy (Sener & Savarese 2018) | each frame's feature is the backbone's feature pyramid, each level averaged over the image, concatenated and L2-normalised; repeatedly the frame farthest (Euclidean) from every frame in training or already chosen |
| `hybrid` | uncertainty, then diversity | the 4 × B frames of highest entropy, then k-centre greedy among them |

Uncertainty ties go to the frame listed first; random draws are seeded by (seed, round).

**Not implemented, and why.** *MC dropout*: torchvision's detection heads have no dropout layers (Faster R-CNN's box head
is two linear layers; FCOS and RetinaNet heads are convolutions with GroupNorm), so it would need a different model
trained with dropout. *Ensembles*: K trainings per round, K times the experiment's cost. *Learning loss* (a loss
predictor): replaced by the k-centre diversity, which needs nothing learned.

## The experiment

```sh
falcon experiment active --workspace W --id ID --pool falcon:RUN... --test falcon:RUN... [--val SRC...] \
    [--strategies random,entropy,kcenter,hybrid] [--initial 32] [--budget 32] [--rounds 3] [--seeds 3] \
    [--aggregate max|mean|sum] [--epochs N --arch A --init I --size WxH --batch B --device D] [--boot N]
falcon experiment results W/experiments/ID          # the table
```

Sources are those of `falcon train` (`falcon:` drives, `version:` dataset versions; a test set may also be a real
set such as `kitti_object:`).

- **The initial set** is the first `initial` frames of a seeded permutation of the pool — the same for every strategy —
  and its model (round 0) is trained once per seed and shared: every curve starts from the same point.
- **Each round** adds B frames chosen by the strategy with the previous round's model, and trains a new model **from the
  same initialisation with the same settings** on all the frames chosen so far (no warm start: rounds differ only by
  their data). At a given round every strategy trains on the same number of frames for the same number of iterations.
- **Every model is scored on the same test set**, which may share no image and no drive (or sequence) with the pool or
  the validation set — otherwise the experiment is refused. With `--val`, each training keeps its best epoch by
  validation AP; without, its last.
- **Resumable**: a finished run is skipped; a round's choice is kept in its run (`selection.json`, with the SHA-256 of
  the model that scored it and the frames in training before it), so a resumed experiment never re-scores. A finished
  training keeps only its chosen weights (no optimizer state), removed once no later choice needs them — round 0's
  after every strategy's first choice (`--keep-checkpoints` keeps them): at most round 0's and one chain's weights
  are on disk at a time (with a training in progress, its checkpoint too). The definition is fixed: the same id
  with another definition is refused.

```
experiments/<id>/
  experiment.json                     the definition and its SHA-256 (core.experiments tracking)
  runs/seed<s>-r0/                    the shared round 0 of seed s: labelled.json, training/ (core.training)
  runs/<strategy>-seed<s>-r<r>/       selection.json (the frames chosen and their scores, the pool's score spread),
                                      labelled.json (every frame in training), training/, params, metrics, tags
  runs/analysis/                      the analysis as a tracked run (MLflow export as for any experiment)
  results.json                        falcon.active_results/1
```

## Honest intervals

The mixed experiment's method ([training](training.md#the-mixed-simreal-experiment)), applied to curves:

- A strategy's **AP at a round** is the mean over seeds of COCO AP on the test set. Its 95 % interval is a percentile
  bootstrap over the test set's units — drives (or a real set's sequences; images only when a set names none) — every
  run resampled together.
- Each strategy against **random at the same round**: the difference on the same resamples (paired), and the verdict
  from the combined interval diff ± 1.96 √(sd_boot² + se² + se_random²), se the standard error of the seed-to-seed
  spread (a normal approximation treating test-sample and training noise as independent). "Better" or "worse" only
  when it excludes zero; otherwise "no detectable difference at 95 %". With one seed the training noise is not
  measured, and the verdict says so.
- **Area under the curve**: AP against frames by trapezoids over the rounds, divided by the frames spanned (a mean
  AP), against random's, paired the same way: the summary verdict ("better than random: more data-efficient").
- **Frames to reach random's final AP**: where the strategy's mean curve first reaches the AP random has at the last
  round (linear between rounds), the share of frames saved, and a bootstrap interval over test resamples (training
  noise not included; "not within the frames tried" when it never does).
- Warnings: fewer than 5 test units (the intervals are not reliable, and a better/worse verdict says so), one seed,
  fewer than 50 scored test objects, AP below 0.01 everywhere (the models detect almost nothing).

`tests/test_activelearn.py` checks the analysis on predictions with known answers (areas 0.25 and 0.75, a saving of a
third, the combined intervals), each strategy's choice by hand (ordering and ties, k-centre greedy, the hybrid's
short-list, the seeded draw), and the entropy and margin of a Faster R-CNN whose class head gives known logits.

## In the loop

Step 5 of the loop ([loop](loop.md)) can choose which generated frames go to training: the ODD's
`"generate": {"acquisition": {"strategy": "entropy", "frames": 400, "aggregate": "max"}}`. After the targeted drives and
the failing-case re-runs are recorded, the model being retrained (a detector Falcon trained: its training folder's
chosen weights, built from its own pinned initial weights) scores all their frames, and only the chosen `frames` go to
step 6 — as training sources `falcon:RUN?select=acquisition.json` (a source option that keeps the listed frames). The
step records `acquisition.json` (the strategy, the model's run.json hash, the frames chosen and their scores) and says
in its decisions how many of how many frames were kept. **Absent by default**, and then every generated frame is kept
as before. With a model Falcon did not train, or fewer frames than asked for, every frame is kept and the step says why.
On CUDA the scoring takes the GPU lock like a training. The step's cost estimate does not include the scoring.

## The page

Training ▸ *Active learning*: the strategies (random always), the initial frames, the budget per round, the rounds,
the seeds, the epochs, the architecture and initialisation, the pool and the held-out test drives (a drive cannot be
in both); *Run the experiment* is a GPU job (`falcon experiment active`). On the right, for the chosen experiment: AP
against training frames, one line per strategy (each with its own colour and marker shape, named in the legend), each
point with its 95 % interval as a whisker; and a table of each strategy's AP at every frame count with its interval,
its area under the curve, the comparison with random and its verdict, and the frames it needs to reach random's final
AP.

## Measured

On the reference machine (RTX 5080; torch 2.9.1 + CUDA 12.8), 2026-10-29, through the GPU lock: the real CARLA 0.9.16
drives on disk — pool: a Town10HD_Opt autopilot drive (160 frames, 614 cars and 90 pedestrians scored) and an imported
Bengaluru drive (120 frames); test: a Town10HD_Opt `lead_brake` drive (240 frames, 379 cars and 74 motorcycles).
Faster R-CNN on MobileNetV3-Small from its ImageNet weights (the COCO detector weights were not downloaded: Falcon never
fetches them by itself), 640 × 384, batch 8, 8 epochs; 20 initial frames, 3 rounds of 20, 3 seeds; random, entropy,
k-centre and hybrid.

| strategy | AP × 10³ at 20 / 40 / 60 / 80 frames (mean of 3 seeds) | area under the curve × 10³ | vs random (combined 95 %) | random's final AP reached at |
|---|---|---|---|---|
| random | 0.06 / 0.01 / 0.22 / 0.19 | 0.12 | — | 57 frames |
| entropy | 0.06 / 0.04 / 0.99 / 0.42 | 0.42 | +0.31 [−0.09, +0.70]: no detectable difference | 43 frames |
| k-centre | 0.06 / 0.03 / 0.65 / 1.36 | 0.47 | +0.35 [+0.11, +0.59]: "better", flagged not reliable (one test unit) | 45 frames |
| hybrid | 0.06 / 0.03 / 1.45 / 0.84 | 0.64 | +0.53 [−0.36, +1.41]: no detectable difference | 42 frames |

Cost: 39 trainings (3 shared rounds 0 and 3 seeds × 4 strategies × 3 rounds) and 27 scorings of the pool took 9 min
52 s on the GPU (290 s of it in the training loops: 14–100 images/s, the first epoch of each training the slowest;
peak 1731 MiB allocated); the analysis, 1000 bootstrap resamples of 39 runs' AP on 240 test images, 4.5 min on the
CPU. Disk: at most about 130 MB of weights at a time (the experiment folder ends at 53 MB).

**This is not a finding about active learning.** With at most 80 training frames from an ImageNet backbone the models
barely detect anything (AP of the order of 10⁻³; the results now warn below 0.01), and the test set is a single drive —
one bootstrap unit, so the intervals over the test sample are degenerate (zero width) and the verdicts rest on the
seed spread alone; a "better" or "worse" verdict on fewer than five test units is now marked not reliable. It measures
the machinery and its cost on real data. A real answer needs a pool of many drives, a test set of five or more held-out
drives (or a real set with sequences), and a detector that starts from COCO weights.

## Limits

- 2D box detectors only, as the trainer ([training](training.md)); one camera per drive.
- Every round retrains from the start: the cost grows with the rounds (rounds × seeds × strategies trainings, the
  round 0 shared per seed).
- The frame is the unit chosen, but frames of one drive are not independent: a strategy may pick near-duplicates
  (k-centre and hybrid counter that; entropy does not).
- Uncertainty is the model's own: a confidently wrong model is not uncertain. The detection-level scores depend on the
  0.3 floor and the score threshold (heuristics).
- The intervals cover the test sample and the seeds; how the pool was recorded (its worlds, weathers, traffic) is not
  resampled, so a finding holds for pools like this one.
