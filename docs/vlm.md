# Vision-language driving explanations

Falcon asks a vision-language model (VLM) grounded questions about a recorded frame, or a short clip ending at it:
*Why should you brake? How many pedestrians are within 50 m? Which lane is the car ahead in? What is the time to
collision with the vehicle ahead?* It then scores the free-text answers against the simulator's exact state, the
facts the [language data](world-model.md#language-data-falconlanguage1) were computed from. No LLM judges the
answers. The rules are written down below, so you can check any score by hand.

Code: `core/vlm.py` (questions and scoring), `core/vlm_eval.py` (the evaluation, `falcon vlm`, the report),
`core/vlm_models.py` (runners, candidate models, download and verification) and `core/vlm_lora.py` (LoRA
fine-tuning). The GUI is the **Language** tab of the Models page.

## Questions and how answers are scored

Each frame's questions come from its `language.jsonl` record. Objects are named by core.language's own definite
phrases: the lead vehicle first, then the nearest described objects (`max_objects`, default 2). Every question goes
to the model with a fixed instruction: *Answer in one short sentence. Give distances in metres, speeds in m/s and
times in seconds. "You" are the ego vehicle whose camera took the image.* The answer is free text.

Before a number, class or side is read from an answer, the answer's object phrases are blanked out. So "the car
12 m away is 14.4 m away" reads 14.4.

| type | example question | scored | rule |
|---|---|---|---|
| `count_vehicles`, `count_pedestrians`, `count_cyclists` | How many pedestrians are within 50 m of you? | exact | the first integer or number word (zero … twenty; no / none → 0; a / an / one → 1); groups are the BEV classes (a motorcyclist is a cyclist) |
| `lead` | Is there a vehicle ahead of you in your lane? | exact yes/no | the first word yes or no; otherwise, in the first clause, a negation → no, an affirmation (there is, I see, you should, brake, stop) → yes |
| `brake` | Should you brake now? | exact yes/no | as `lead`, against core.language's heuristic brake rule |
| `lane` | Which lane is the vehicle ahead of you in? | exact class | your lane / n lanes to the left or right (n counted) / oncoming / crossing / off the roadway; only asked on drives with lanes.json |
| `relative_position` | Is the car 12 m away on your left or on your right? | exact | the last left/right word, or ahead/behind for an object within 1 m of the ego's axis |
| `motion` | Is the car … getting closer or moving away? | exact class | closing / pulling away / standing still / keeping its distance, using core.language's thresholds |
| `distance` | How far is the car ahead on your right from you? | tolerance band | correct when \|answer − truth\| ≤ max(2 m, 15 % × truth); truth is the box-to-box gap |
| `speed` | How fast is … moving? | tolerance band | max(1 m/s, 15 %); km/h and mph are converted; "standing still" counts as 0 |
| `ttc` | What is the time to collision with the vehicle ahead of you? | tolerance band | max(0.5 s, 20 %). A truth of "no collision course within 10 s" is matched by an answer saying so, or by a number at or beyond the horizon |
| `brake_reason` | Should you brake now? Say why. | faithfulness | below |

The bands are heuristic: wide enough for a monocular estimate to count, narrow enough that a guess does not.
`scores.json` states them, and `--tolerance TYPE=ABS,REL` changes them (for example `--tolerance distance=1,0.1`). Each numeric answer's
absolute error is kept, along with the median error and its interval.

The lead's lane is "your lane" by core.language's own definition of the lead, so that question tests whether the
model agrees that the vehicle ahead is in its lane. The lane questions about other objects are the informative ones.

### Grounding: which object an answer refers to

A **mention** is a determiner (the, a, an, that, this, or a count), an optional ordinal (nearest, second-nearest …),
up to two other words and an object noun (car, truck, bus, pedestrian, person, child, cyclist, motorcyclist, cone,
animal …). The words after the noun belong to the mention as long as they are qualifier words: a distance ("12 m"),
a direction, a lane ("in your lane", "in the oncoming lane"), "track id N" or "ahead of you". The mention ends at
"N m away". The mention is resolved against the frame's actors within the language data's radius (50 m), in this
order:

1. "track id N" names that actor.
2. **Candidates** are the actors whose group the noun fits. Vehicle nouns fit vehicles, and "vehicle" also fits
   cyclists. Person nouns fit pedestrians, bicycle nouns fit cyclists, and "cone" or "animal" fit no actor. Candidates
   with the mention's own core.language noun come first when there are any, so "the police car" prefers police cars.
3. A bare definite mention ("the car", with no distance, direction, lane or ordinal) **refers back to the object the
   question named**, when that object is a candidate. After that, "the vehicle ahead (of you)" with no distance or
   ordinal is the lead vehicle when there is one.
4. A stated direction keeps the candidates in its sector. Falcon's own words (ahead, ahead on your left, on your
   right, behind on your left, behind …) use their sectors exactly, then widened by 15°. Free words use half-planes:
   left is y < 0, right y > 0, ahead x > 0 and behind x < 0, in CARLA's frame where +y is right. A stated lane keeps
   the candidates in that lane relation.
5. A stated distance keeps the candidates whose gap or centre range is within the default distance band (whatever
   band the scoring uses), and picks the nearest in gap. An ordinal picks the k-th nearest. Otherwise the nearest candidate is picked, flagged as ambiguous
   when there were several.

A mention that resolves to nothing is a **hallucination**, unless its stated distance lies beyond the radius (then
it is not scored). A count mention ("2 cyclists") is a group claim. It is a hallucination only when no actor of the
group is present.

An answer about an object is **grounded** when its first mention resolves to that object. For brake answers, the
first mention must resolve to one of the reason actors. An answer that names no object is not scored for grounding.
Answers that repeat the question's own phrase are grounded by construction, so grounding is informative mainly for
`lead`, `brake` and `brake_reason`, where the model has to name the object itself.

The **hallucination rate** is hallucinated mentions divided by all mentions. The share of answers with at least one
hallucination is reported too. These are the instance-level and sentence-level forms of CHAIR (Rohrbach et al. 2018,
*Object Hallucination in Image Captioning*, EMNLP).

### Faithfulness of a brake explanation (rule-based)

The answer is cut into clauses at `. ; : ! ?` and at *and, but, because, since, so, as, while*. A clause is an
**absence** claim when a negation word (no, not, nothing, none, nobody, no one, never, without, n't, clear) comes
before its first hazard cue. Otherwise it is a **presence** claim. The hazard kinds are:

- **ttc**: a vehicle mention, or a collision cue (collision, collide, crash, hit, too close, closing, rear-end,
  tailgating, cutting in, obstacle).
- **path**: a pedestrian or cyclist mention.
- **light**: a traffic light or signal, or a red, yellow or amber light. A green light counts as an absence claim.
- **other**: a stop sign, speed bump, junction, turning, weather, an animal and the like. The facts cannot confirm
  these, so they are never supported.

Actions on their own (brake, slow down, stop) are decisions, not reasons.

The true reasons are core.language's heuristic rule, recomputed from the facts: a time to collision ahead below
3 s, a pedestrian or cyclist in or entering the path, or a red or yellow light when one was recorded.

- A presence claim is **supported** when a true reason of its kind exists. If the claim names an object, one of its
  candidates must be that reason's actor. A claim about a hallucinated object is never supported.
- An absence claim is supported when no true reason of its kinds exists. A claim that names no kind covers all of
  them.

**Precision** is supported claims divided by claims. **Recall** is the share of true reasons covered by a supported
presence claim. When there is no true reason, recall is 1 if a supported absence claim was made. An explanation is
**faithful** when it makes a claim, its precision is 1 and its recall is above 0. The decision the explanation
implies (brake if any presence claim) is scored as well.

**An optional judge.** A judge is a client you write and approve: an object with `.complete(prompt)` or a callable,
given as `--judge module:Class --approve-judge`. Falcon opens no connection of its own and never uses a judge by
default. Without `--approve-judge` the evaluation is refused. The judge's FAITHFUL or UNFAITHFUL verdict is stored
beside the rule's, and `scores.json` reports its faithful rate and how often it agrees with the rules. The rules
remain the score.

### Intervals

Every number is a mean of per-answer scores. Each has a 95 % percentile bootstrap interval (`core.metrics.bootstrap`;
Efron & Tibshirani 1993) over **whole drives**: a cluster bootstrap (Davison & Hinkley 1997, §3.8), because frames
of one drive are not independent. With fewer than five drives the unit becomes a **2 s block** of consecutive frames
of a drive, and the scores say that the interval is too narrow, since blocks of one drive are still correlated. The
model harness gives the same caution.

## Runners

| runner | what |
|---|---|
| `oracle` | core.language's own template answer: perfect by construction, and **not a model**. It reads the truth, and it exists to validate the scorer end to end |
| `noisy[:p=0.3,h=0.1,seed=0]` | The template answer, replaced with probability p by an answer that is wrong by construction (`core.vlm.corrupt`). With probability h, its object phrase is replaced by one naming an absent object, a planted hallucination. Seeded per question and **not a model**: its expected scores are known exactly, and each answer's `note` says what was done to it |
| `plugin:module:Class[,k=v]` | Your class with `answer(images, prompt, meta) -> str`. Its `meta` carries the drive, step, frame and question type, never the truth. A `licence` attribute is reported (AGPL is refused) |
| `hf:NAME` / `hf:DIR[,adapter=DIR,device=cpu,max_new_tokens=64,max_side=768]` | A local Hugging Face model run with torch through transformers (AutoProcessor and AutoModelForImageTextToText, greedy decoding, seeded), loaded with `local_files_only` and `HF_HUB_OFFLINE=1`. NAME is a downloaded candidate, checked against its pins before every load. DIR is a folder, whose licence may be given in a `falcon_vlm.json`. An adapter is a LoRA (below) |

The images of a question are the frame's camera image, or with `--clip K` the K frames ending at it, `--clip-step`
steps apart, oldest first.

## Candidate models: licences, sizes and pins

**No weights are bundled, and nothing downloads by itself.** The candidates are small open VLMs that transformers
runs natively (no `trust_remote_code`). Each licence was read from the model card, both its YAML `license` field and
its text. Each file list, size and published checksum was read on 2026-10-29 from the Hugging Face API's metadata
(`https://huggingface.co/api/models/REPO?blobs=true`), **without downloading any weights**. Each candidate is pinned
to a repository revision (commit).

| name | repository @ revision | licence | parameters | download |
|---|---|---|---|---|
| `smolvlm-256m` | HuggingFaceTB/SmolVLM-256M-Instruct @ 7e3e67ed | Apache-2.0 | 0.26 B | 0.48 GiB |
| `smolvlm-500m` | HuggingFaceTB/SmolVLM-500M-Instruct @ a7da5b98 | Apache-2.0 | 0.51 B | 0.95 GiB |
| `llava-onevision-0.5b` | llava-hf/llava-onevision-qwen2-0.5b-ov-hf @ 74dd0bf8 | Apache-2.0 | 0.89 B | 1.68 GiB |
| `qwen2-vl-2b` | Qwen/Qwen2-VL-2B-Instruct @ 895c3a49 | Apache-2.0 | 2.21 B | 4.13 GiB |
| `smolvlm-2.2b` | HuggingFaceTB/SmolVLM-Instruct @ 81cd9a77 | Apache-2.0 | 2.25 B | 4.19 GiB |
| `smolvlm2-2.2b` | HuggingFaceTB/SmolVLM2-2.2B-Instruct @ 482adb53 | Apache-2.0 | 2.25 B | 8.37 GiB (float32) |
| `granite-vision-3.2-2b` | ibm-granite/granite-vision-3.2-2b @ ec894191 | Apache-2.0 | 2.98 B | 5.55 GiB |

Parameter counts are the API's safetensors totals. The two SmolVLM 256M and 500M models are below the 1–4 B range.
They are listed as the cheapest first tries. Papers: SmolVLM (Marafioti et al. 2025, arXiv:2504.05299), Qwen2-VL
(Wang et al. 2024, arXiv:2409.12191), LLaVA-OneVision (Li et al. 2024, arXiv:2408.03326), Granite Vision (IBM 2025,
arXiv:2502.09927).

**Refused by default**, each with its reason as read on 2026-10-29:

| model | licence | why |
|---|---|---|
| Qwen/Qwen2.5-VL-3B-Instruct | qwen-research | non-commercial: its LICENSE defines "Non-Commercial" as research or evaluation only |
| apple/FastVLM-1.5B | apple-amlr | research only; also needs trust_remote_code |
| google/paligemma2-3b-pt-224 | Gemma Terms of Use | not an open licence (use restrictions), and gated |
| deepseek-ai/deepseek-vl-1.3b-chat | other (deepseek) | cannot be checked: the card links a LICENSE the repository does not hold; the DeepSeek licence carries use restrictions |
| openbmb/MiniCPM-V-2 | not stated in the card's metadata | commercial use only after registering with the publisher; trust_remote_code |
| microsoft/Phi-3.5-vision-instruct | MIT | needs trust_remote_code: running a repository's Python is refused |
| vikhyatk/moondream2 | Apache-2.0 | needs trust_remote_code |
| OpenGVLab/InternVL2_5-2B | MIT | needs trust_remote_code |

No AGPL-licensed VLM was among the models checked. Falcon would refuse one, as `core.models` does.

**How the hashes are pinned.** For each file the metadata states the publisher's checksum. For an LFS file (the
weights) this is its SHA-256. For a small file it is its git blob id: SHA-1 of `blob <size>\0` followed by the
content. Falcon has not downloaded the files, so its own SHA-256 pins are marked **"to be pinned on first approved
download"**. The download works like this:

1. `falcon vlm fetch NAME --workspace W` prints the repository, revision, licence, sizes and every file's published
   checksum, then asks for confirmation. Typing `yes` confirms, or `--yes` if you have read the text. With no
   terminal and no `--yes`, the fetch is refused with exit code 6.
2. Every file is downloaded from `https://huggingface.co/REPO/resolve/REVISION/FILE` into `W/models/vlm/NAME/`. Each
   file's size and published checksum are checked; on a mismatch the file is deleted and the download refused.
3. Falcon computes each file's SHA-256 and writes it to `pins.json` beside the files. That first approved download
   is the pin.
4. Every later load (`hf:NAME`) and `falcon vlm verify NAME --workspace W` check every file against `pins.json` and
   the publisher's checksums.

To re-read the metadata yourself without downloading weights:
`curl -s "https://huggingface.co/api/models/REPO?blobs=true"`. The `siblings` list has each file's `size`, its
`blobId` and, for LFS files, `lfs.sha256`. `sha` is the revision.

## LoRA fine-tuning

`falcon vlm finetune --model NAME|DIR --lang DIR… --out DIR` trains a LoRA adapter on Falcon's grounded QA.
LoRA (Hu et al. 2021, arXiv:2106.09685) keeps the base weights frozen and adds a low-rank update
y = W x + (α / r) B A x to selected linear layers. A starts Kaiming-uniform and B starts at zero, so training starts
from the base model exactly.

- **The LoRA is Falcon's own**, about 60 lines of torch. peft 0.20 is installed on the reference machine, but Falcon
  does not use it or depend on it.
- **Targets** are the language model's attention projections by default: `q_proj, k_proj, v_proj, o_proj`, rank 8,
  α 16, dropout 0.05. The image encoder is left alone (`--scope text`); `--scope all` adapts it too.
- **The data** are (images, question, core.language's template answer) examples from the language folders.
- **Train and validation are split by drive.** The drives are put in a seeded order and the last ⌈25 %⌉ (at least
  one) validate. Fewer than two drives is refused.
- **Training.** The loss is cross-entropy on the answer's tokens only. AdamW on the LoRA parameters, with warm-up
  then cosine, gradient accumulation and clipping at a norm of 1.
- **Each epoch** records the validation loss and core.vlm's scores of the model's own greedy answers on up to
  `val_generate` validation questions. The kept adapter is the epoch with the lowest validation loss.
- **Resume.** A cancel saves a checkpoint at the last optimiser step. A resumed run replays the partial accumulation,
  and on the CPU it gives the same adapter as an uninterrupted run (tested).
- **Outputs**: `adapter/adapter.pt` and `adapter/adapter.json`, plus `metrics.jsonl`, `split.json` and `run.json`.
  `adapter.json` records the base model and its licence, because an adapter is a derivative of its base model.

**Real fine-tuning awaits your approved download of a candidate's weights.** The code is tested on a tiny,
randomly initialised Idefics3 (SmolVLM's architecture, about 70 000 parameters) whose tokenizer is trained in the
test. Nothing in the test is downloaded.

## Outputs

```
vlm/evaluations/<time>-<runner>/        (the Language tab's folders; any --out DIR from the CLI)
  answers.jsonl   one line per question: drive, step, frame, question, answer, truth, target, template, images,
                  latency, the score (parse, verdict, mentions and their resolution, claims), the runner's note
  scores.json     per question type: accuracy, grounding, median error, explanation precision/recall/decision,
                  share parsed, each with its interval; macro accuracy, grounding, hallucination rate; the method
                  (bands, rules, prompt); the runner with its licence; the judge if one was approved
  run.json        provenance: the language files' SHA-256, the model's pins.json or adapter, settings, seed, versions,
                  timings, status; the outputs' hashes
  language/<drive>/   language data derived from --run drives (template text only)
```

An evaluation **resumes**: answers are written as they come, and the same command continues a cancelled run. The
same folder with other settings is refused. `falcon report vlm DIR --out FILE` writes the HTML report.

## Commands

```sh
falcon vlm list [--workspace W]                      # candidates with licences, sizes, state; the refused ones
falcon vlm fetch NAME --workspace W [--yes]          # licence and checksums shown, asks first
falcon vlm verify NAME --workspace W
falcon vlm eval --runner oracle|noisy[:p=…,h=…,seed=…]|plugin:M:C|hf:NAME[,adapter=DIR] \
                (--lang DIR… | --run RUN…) --out DIR [--every 5] [--max-frames N] [--max-objects 2] \
                [--clip K --clip-step S] [--types T,…] [--tolerance TYPE=ABS,REL] [--boot 1000] [--seed 0] \
                [--judge M:C --approve-judge]
falcon vlm finetune --model NAME|DIR --lang DIR… --out DIR [--epochs 2 --lr 2e-4 --r 8 --alpha 16 --accumulate 8 …]
```

An `hf` runner on CUDA is a `needs_gpu` job (`core.vlm_eval.job_spec`). Run from the CLI, it takes core.training's
GPU locks: it waits while a CARLA job holds the GPU, and holds the CARLA lock so CARLA jobs queue behind it.

**The Language tab** (Models page) lists the candidates and downloads one on request, after a dialog shows its
licence and checksums. It runs an evaluation as a job on the checked drives, deriving their language data, with the
oracle, the noisy baseline, any downloaded candidate or a plug-in. It shows the per-type scores with their intervals,
and lets you browse frames with each question, answer, truth and verdict.

## Measured

The scorer's own consistency, on the two real Town10HD runs recorded on CARLA 0.9.16 (an 8 s drive and a lead_brake
scenario; language data derived here):

- **Resolution.** The resolver maps **9,913 of 9,913** single-object phrases that core.language wrote (captions,
  questions, answers) to the track ids core.language attached, plus 200 group phrases. This shows the rule agrees with
  Falcon's own referring expressions. It is no measure of how well it reads a model's wording.
- **The oracle**, over all 400 records and every question type: **5,906 of 5,906** questions correct, every answer
  grounded, no hallucination.
- **An evaluation at every 5th frame**: 80 frames and 1,071 questions; intervals over 2 s blocks, because two drives
  are too few for drive-level intervals.
  - **The oracle** scores **1.000 [1.000, 1.000]** on every type, with grounding 1.000 and hallucination rate 0.000.
  - **The noisy baseline** (p 0.3, h 0.1, seed 0) has macro accuracy **0.720 [0.703, 0.738]**. Per type it ranges
    from 0.651 (speed) to 0.762 (lead, count_pedestrians). Explanation precision is 0.753 and recall 0.762, grounding
    0.864 [0.824, 0.898] and hallucination rate **0.183 [0.159, 0.213]**. All 1,071 answers scored exactly as the
    corruption log predicts: a corrupted answer is wrong, an untouched one right, and a planted hallucination is
    found. Each evaluation took 0.33 s on this machine, including scoring and 1,000 bootstrap resamples.
- **The tests** (`tests/test_vlm.py`) calibrate the scorer with 47 hand-written answers whose scores are known.

No VLM has been scored yet, because none has been downloaded.

## Limits

- **Only actors are objects.** Town10's parked cars are map meshes, not actors, so a model that correctly names one is
  counted as hallucinating. The same goes for objects beyond 50 m that the answer gives no distance for.
- **The brake rule and lane relations are core.language's heuristics.** Traffic-light states are not recorded by
  default, so the light reasons rarely apply.
- **The parsing is English-only and keyword-based.** A correct answer phrased unusually can score wrong; unparsed
  answers are counted and shown per type. Nouns are checked at group level only: a "truck" that is a car still
  grounds.
- **The tolerance bands are choices, not standards.** Change them and the accuracies change; the errors are kept so
  you can apply other bands.
- **Grounding is informative mainly where the model must name the object itself.** An answer that repeats the
  question's phrase is grounded by construction.
- **Intervals cover the frames and drives sampled, not other towns or weathers.** With fewer than five drives they
  are too narrow, and the scores say so.
- **The HF runner and LoRA are tested only on a tiny random model.** On a real model, the chat template, image
  resolution and memory use are the publisher's; some processors may tokenise prompt + answer so that the prompt is
  not a prefix, which fine-tuning refuses.

## References

Hu et al. 2021, LoRA, arXiv:2106.09685 · Rohrbach et al. 2018, Object Hallucination in Image Captioning (CHAIR),
EMNLP, arXiv:1809.02156 · Efron & Tibshirani 1993, An Introduction to the Bootstrap · Davison & Hinkley 1997, Bootstrap
Methods and their Application · Related driving QA benchmarks: DriveLM (Sima et al. 2023, arXiv:2312.14150),
NuScenes-QA (Qian et al. 2024, arXiv:2305.14836), BDD-X textual explanations (Kim et al. 2018, ECCV,
arXiv:1807.11546).
