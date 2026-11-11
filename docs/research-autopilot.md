# The research autopilot

The **research autopilot** takes a question — *does the fine-tuned detector raise night recall for two-wheelers?* —
through the whole research loop: an ablation designed with a power analysis, approved by you, run through the job
queue, analysed, followed up when it was inconclusive for want of power, and written up as a report whose every number
is checked against what Falcon measured. The assistant's model writes the ablation spec and the discussion; a
deterministic orchestrator in Falcon owns everything else — the order of the steps, which step asks you, the waiting,
the follow-up rule, the report's numbers and its check, and resuming after a restart.

Code: `falcon_autonomy/core/autoresearch.py` (the orchestrator), `assistant/tools/autopilot.py` (the assistant's
`research_autopilot` and `research_studies`), `ui/research_view.py` (the Assistant panel's **Research** tab). Each
round is an ablation of the experiment registry ([experiments](experiments.md)).

## Starting a study

- **Ask the assistant** in any permission mode: *"Make this a research study: does the fine-tuned detector raise
  night recall?"*. It writes the spec — factors and levels, the metric, the contrasts, the effect to detect and the
  spread of paired differences — and calls `research_autopilot` with `action: start`. Nothing is written until you
  approve the card.
- **Pick a suggested question**: Assistant panel ▸ Research ▸ *Find* lists questions found in the project's own state,
  each with *Study this* (it goes to the assistant as a request). The assistant can read the same list
  (`research_studies` with `suggest`).

| source | the question | the template spec |
|---|---|---|
| a failure mode ([failure clusters](failure-clusters.md)) | does perception degrade in its conditions? | its time of day, fog, rain or sun as a weather level against a clear day |
| an operating-domain cell above the failure target, or with too few runs ([ODD coverage](odd-coverage.md)) | is perception worse in that cell? | the cell's representative weather and sun against a clear day |
| a failure-rate estimate's worst cell ([failure rate](failure-rate.md)) | does perception explain the rate there? | as for a cell |
| a regression a bisect found ([bisect](bisect.md)) | what did that commit change? | none: commits are not a factor the ablation executors run |

A template compares the project's first two registered models in the condition (or, with one model, the condition
against a clear day); its effect 0.05 and spread 0.06 are the manual's example unless an analysed ablation of the
workspace measured the spread for the same metric — the suggestion says which. The road, actor and speed of a cell are
not ablation factors: the template varies weather and light only, and says so.

## The approval card

The first card shows the question, the hypothesis, every factor with its levels, the plan (cells × replicates = runs,
the seeds, each contrast's pairs and power from the power analysis), the cost (for the detection executor: its drives
at the rate measured on this machine, *typical* until a drive measured it; evaluations are counted, not timed) and
what happens after. It has **Approve** and **Reject** — no *Approve all of this kind* — and one box: **Approve the
follow-ups of this study** (more replicates or a narrower factor, within this plan's factors and levels, without asking
again; Stop ends it).

## Approvals in each permission mode

Every step of a study that queues compute asks you on a card — in Autopilot too. These are the user's rules for this
loop; the orchestrator checks them itself, so a step that queues compute without your approval is refused whoever
calls it.

| step | Ask | Auto | Autopilot | approved ahead (the first card's box) |
|---|---|---|---|---|
| start (register the plan, queue its job) | card | card | card | — (this is the first card) |
| follow_up (a new round) | card | card | card | no card, within the approved factors and levels |
| resume (after Stop, a failure or a restart) | card | card | card | no card — unless the study was stopped |
| resume when every run had finished (only the analysis is written) | card | runs | runs | runs |
| report | card | runs | runs | as the mode says |
| stop | card | runs | runs | as the mode says |
| read a study, list the suggestions | runs | runs | runs | runs |

There is no budget: nothing caps the rounds or the replicates beyond the spec's own `max_replicates`. **Stop** ends a
study: the Research tab's Stop (it also stops the assistant's turn), the panel's Stop while it waits for the study's
job, or Undo in the activity log (it cancels the job; the study then reads as stopped). A Stop revokes the follow-ups
approved ahead: resuming afterwards asks again.

## The state machine

The study is `experiments/<id>/autopilot.json` beside its first experiment; its rounds are the experiments `<id>`,
`<id>-f1`, `<id>-f2`, …. The file holds the decisions — each round with its spec, plan, approval (how and when) and
jobs, a stop, the report — and the history of the states. The state itself is worked out from the decisions, the
experiments' files and the job queue, so a study resumes by construction after a restart.

| state | means | next |
|---|---|---|
| proposed | the plan is drafted; its card is open (nothing written yet) | approve the card |
| approved | registered; its job not queued yet | resume queues it under the same approval |
| running | its job is queued or running in this session | wait (`plan_wait_jobs`; Autopilot waits by itself) |
| interrupted | its job is gone — Falcon was restarted — or ended without finishing every run | resume (finished runs are not repeated) |
| failed | a run failed (the experiment's run says why) | resume runs it again from there |
| analysed | every run finished and was analysed: conclusive, or inconclusive with no follow-up | the report |
| follow_up | analysed, inconclusive for want of power: a follow-up is proposed | follow_up (a card), or the report |
| reported | the report covers every round | — (a draft: write it again) |
| stopped | you stopped it | resume, if you ask |

A job is known to a study by its id and its submission time, so a job of an earlier session is never taken for one of
this session's (job ids start again at each start of Falcon).

## The follow-up rule

A round is **inconclusive** when its hypothesis is *not supported* and no contrast contradicts it. A follow-up is
proposed only when an inconclusive contrast was underpowered: the power the round had for the planned effect at the
spread it observed is below the target power, or every bootstrap resample went one way (the p-value's floor). A round
that had the power and found nothing is an answer — *no detectable effect of the planned size* — and gets no
follow-up.

The follow-up is a fresh experiment with new seeds (the round before's first seed + 1000), planned for the planned
effect at the largest spread observed among the underpowered contrasts, with at least one replicate more than the
round before. When the hypothesis was split by a factor (`by`) and some of its contrasts were supported while others
were not, it narrows that factor to the inconclusive levels. Rounds are not pooled. **Each look halves alpha** (round k
at alpha / 2^(k−1)), so the chance of a false *supported* over all the looks of a study stays below twice the first
round's alpha — a Bonferroni bound over the looks; the report names the look that reached the verdict.

## The report

`research_autopilot` with `action: report` writes `experiments/<id>/report.md` and `report.json`, and each round's
outputs for a paper into `paper/<experiment>/` (LaTeX and CSV tables, PDF, PNG and SVG figures). Falcon writes:
the question and its source, the hypothesis, the design (factors and levels, executor, each round's plan, power
analysis and approval), the pins of every round, every contrast with A and B, the difference with its 95 % and
simultaneous intervals, dz with its interval, the adjusted p-value and the verdict, the study's verdict and the look
that reached it, the limits (each round's warnings, the assumed spread, *no detectable effect* is not *no effect*,
what the executor varies) and the paper files. The model's discussion goes in its own section.

**Every number is checked** with the assistant's grounding check against the study's own facts: the record, each
round's plan summary, spec and results, the proposed follow-up. The per-run list and the list of seeds are left out of
the facts, so that a run index or a seed cannot ground an invented number. A number no fact holds is marked *[not from
Falcon's data]* where it stands and listed at the end; such a report is a **draft**, never final. The assistant is
told to write the discussion again with the study's own numbers.

## Privacy

Suggestions never come from the dashcam footage: failures, runs and studies the privacy rules call private
(`assistant.privacy`) are left out before anything is clustered, counted or estimated, so a private failure shapes no
cluster, no cell and no suggestion. `start` refuses a source that is not one of the suggestions (a private cluster id
cannot be one) and a spec whose world is an import of a dashcam route. A study's folder is read through the privacy
guard.

## The Research tab

Assistant panel ▸ **Research**: every study with its state, rounds and verdict (a study whose first card is open shows
as *proposed*); the selected study's state machine as one line (*proposed → approved → [running] → analysed →
follow-up → reported*), its rounds (experiment, runs, status, verdict, how each was approved, its jobs), whether its
follow-ups were approved ahead, and its history. **Stop**, **Undo** (the newest change the assistant made for the
study, through the activity log), **Resume** (its card, as the rules above say) and **Open report**; and the suggested
questions with *Study this*.

## For developers

- `core.autoresearch`: `draft`, `start`, `follow_up_proposal`, `follow_up`, `resume`, `stop`, `report`, `status`,
  `derive`, `sync`, `within_bounds`, `preapproved`, `suggestions`, `facts`, `ground`. The job queue is reached only
  through callables the caller passes (`submit(exp_id) -> {'id', 'submitted'}`, `job_state(job record)`,
  `cancel(job record)`); the approval it is handed (`{'how': 'approved' | 'preapproved' | 'command line', 'options'}`)
  is checked for every step that queues compute.
- `tool_api.Tool.approval(ctx, **args)`: a tool's own say on its card — `None` (the permission mode decides),
  `{'ask': True, 'allow_all', 'options', 'why'}` (a card in every mode) or `{'ask': False, 'why'}` (approved ahead, no
  card). The agent puts how a call was approved, and the boxes ticked, in `ToolContext.approval`; `falcon assistant
  call --allow-act` sets `{'how': 'command line'}`.
- The study's job is `falcon experiment ablation run <id>` (`--workspace` for the simulated executor, `--project` for
  the detection executor), with `JobSpec.meta = {'study', 'experiment'}`.

## Limits

- Ablations vary what the executors run: models, weather, worlds, rigs and traffic (detection), or stated values
  (simulated). A question about a scenario's parameters, a sensor setting or a commit needs another executor; the
  suggestions say when a template cannot express it.
- The follow-up rule is one rule: more replicates at the observed spread, or a narrower `by` factor. It does not
  propose new factors; a broader question is a new study.
- Halving alpha keeps the error over all looks below twice the first round's alpha, not at it; rounds are separate
  experiments and are not pooled, which spends data for that guarantee.
- The grounding check finds numbers no fact holds; it does not prove the discussion right (a correct number can be
  cited for the wrong claim).
- The cost estimate times the drives from the measured throughput; evaluations are not timed.
- Measured with the scripted model (tests/fake_groq.py) and the simulated executor only: the whole loop — start,
  an inconclusive round of 16 runs, a follow-up of 48, a final report — in each permission mode, the resume after a
  restart in the middle of a round, Stop, the draft report, the private failure mode; one study job of 16 simulated runs
  through the real job queue as a subprocess took 0.12 s on the reference machine (2026-11-11). Not yet run against
  the real model (how well it writes specs and discussions, and whether it follows the next steps) or with the
  detection executor on a real CARLA server.
