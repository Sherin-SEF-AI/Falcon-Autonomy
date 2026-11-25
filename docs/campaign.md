# Overnight campaign with a morning brief

Give Falcon a goal and a budget in the evening —

```
falcon campaign new my.falcon.json "Find why night pedestrian recall dropped" --gpu-hours 6 --until 07:00
falcon campaign approve my.falcon.json c-4be1a9f0
falcon campaign run my.falcon.json c-4be1a9f0
```

— and read a one-page brief in the morning: the answer in two sentences with its confidence, what was tried, what was
ruled out, the evidence, the budget used, what needs you, and the next step.

A campaign is a layer **on top of** what Falcon already has, not a second autopilot. It chooses which existing study
to run next, runs it as the command the pages and the CLI already run, reads its result files and writes down every
decision. An ablation goes through the [research autopilot](research-autopilot.md) with its power analysis.

## What a campaign is

| Part | What it holds |
|---|---|
| goal | one sentence in your words |
| budgets | GPU-hours, a wall-clock deadline, a job count, disk |
| scope | the study kinds it may run, fixed when you approve |
| plan | hypotheses ranked, each with the study that tests it and its estimated cost |
| log | `log.jsonl`, append-only: every decision with its reason |
| brief | `brief.md` and `brief.html` |

All of it is in `<workspace>/campaigns/<id>/`.

Study kinds: `clusters` (failure modes), `moments` ([what happened in the runs](moments.md)), `rootcause` (which
attributes an evaluation's misses go with), `ablation` (a controlled comparison), and, when you allow them,
`failrate` and `search` (of a project scenario the goal names; both drive CARLA). The default scope is the first
four. Sweeps, falsification and model evaluation are not campaign kinds: the playbook has no rule that could plan one
from a goal alone.

## One approval

`falcon campaign new` writes a draft and prints its whole scope: the goal, the budgets, the allowed kinds, the
planned studies in order with their estimates, and what is never done. **Nothing runs.** `falcon campaign approve`
prints it again and asks once; in the window it is the **Approve and start** button of *Loop ▸ Overnight campaign*;
through the assistant it is one approval card (`research_campaign_new`), asked in every permission mode, Autopilot
included.

No new permission exists for this. The approval is the same object the research autopilot takes, stored in the
campaign and handed to the autopilot unchanged for an ablation. The approved scope is hashed: a campaign whose goal,
budget, kinds or planned commands are edited afterwards does not run.

Inside the scope it runs unattended. Outside it, nothing is done; it is written to the brief as **needs you**:

- a study of a kind you did not allow;
- a study whose estimate alone exceeds a budget;
- anything a planner asks for beyond the candidate studies;
- always, whatever was approved: cloud spend, a model download, moving anything to the trash, training, settings,
  pushing anything, and everything on the assistant's never-list.

## The accountant

Budgets are enforced by code that no model can talk to. Before a job starts it is refused when

- the job count is at its limit;
- its estimate would take the GPU-hours used past the budget;
- its estimate would end after the deadline;
- the disk budget is used up, or less than 1 GB is free on the workspace's disk.

While a job runs it is cancelled, through the job queue's own cancel path, at the deadline or when the GPU-hours are
used up. A campaign that stops, crashes or is interrupted cancels its jobs on the way out; the next `falcon campaign
run` resumes from the files, and a study whose job was lost in a restart runs at most once more.

What is measured and what is assumed:

- **GPU-hours** are the wall time of the studies of a GPU-holding kind — an upper bound on GPU time, not a
  measurement of utilisation.
- **Estimates** are the median duration of finished jobs of the same kind in this workspace's job history. An ablation
  that drives CARLA uses its own estimate (its drives at this machine's measured rate). With no history, a typical
  value per kind is used and labelled "typical, not measured" — the first campaign on a machine plans with rough
  numbers, later ones with measured ones.
- **Disk** is the size of the studies' own output folders.

## The loop

1. **Plan.** A fixed playbook builds candidates from the workspace's public facts: the failure modes of the failures
   recorded, the goal's conditions as moments, the root cause of the two newest evaluations (when the goal is about
   detection quality), the ablations the research autopilot suggests (ranked up when their weather and light are the
   goal's), a failure-rate study and a rare-case search of a project scenario the goal names. They are ordered by a
   stated heuristic: cheap
   and descriptive first, then the controlled comparison, then the long simulator studies.
2. **Run** the best hypothesis still open, if the accountant allows it.
3. **Triage** its result from its own files, and plan again (a result can suggest a new study; it runs only if its
   kind is allowed and it fits what is left).
4. **Stop** when a controlled comparison supported a hypothesis (the goal is answered), at the deadline, when the
   budget is used up, when nothing is left worth its cost, or when you say so.

Two planners:

- `--planner playbook` (the default): the rule-based order above. No model is involved anywhere.
- `--planner assistant`: the configured assistant provider is shown the goal, the budgets and the candidates'
  one-line texts and returns their **order**. It cannot add a study; anything else it asks for becomes a "needs you"
  item. A reply that cannot be read leaves the playbook's order, and the log says so. The accountant, the triage and
  the brief are the same code in both modes.

## Running it

- `falcon campaign run WS ID` runs until it is finished; Ctrl-C (or a `SIGTERM`) cancels its job and exits, and
  running it again resumes.
- `falcon campaign run WS ID --detached` prints a `systemd-run --user` command that keeps it running after the
  terminal closes. Falcon prints the command; it does not run it or install a timer.
- `falcon campaign status WS ID`, `stop`, `brief`, `list`.
- In the window, *Loop ▸ Overnight campaign*: the goal, the budget fields, the scope checklist, **Preview**,
  **Approve and start**, then the budget meters, the decision log and the brief. The runner is a job in the Jobs dock.
  Since 4.2 the preview is compact: the scope and budget as key-value badges (GPU-h, until, jobs, disk, kinds,
  planner, the planned studies with their estimated time and GPU-hours, needs-you), the planned studies as a table
  (each one's hypothesis and the estimate's source in tooltips) and the approval's whole text — exactly what the one
  approval shows — behind *Details*; what is never done unattended is one line with its list behind *Details*. The
  campaign's fidelity is a chip beside its state.
- A desktop notification is sent when it finishes (the same path as the cloud alerts; never in a headless session).

## The brief

One page: the goal; the answer in two sentences and its confidence —

| Confidence | Means |
|---|---|
| high | a controlled comparison with enough power supported it |
| moderate | the recorded evidence points one way, but no controlled comparison confirmed it |
| low | only an association in the data recorded so far; nothing was tested |
| none | no study produced evidence for or against |

— its **fidelity** (4.2: `core.fidelity.combine` over the studies' results, each result recording what it read — a
search's or failure-rate study's summary, else its rows' runs; the moments matched; a root-cause file; `unknown` when
the files do not say) — then what was tried (study, cost, fidelity, result), what was ruled out, studies planned but
not run and why, the evidence
(paths to the studies' folders and the decision log), budget used against given, the needs-you items and one
recommended next step chosen by rule.

Every sentence is assembled from the result files of the studies that ran; the brief contains no text written by a
model. Runs derived from dashcam footage are not read: the studies behind it use public sources only and report
private ones as a count.

## Limits

- **A conclusion is only as good as the studies that ran.** Only an ablation tests a cause; failure modes, moments and
  root cause describe associations in what was recorded. The brief says which it is.
- The playbook is short and fixed. It does not invent experiments; a goal none of its rules bears on ends with "did
  not find an answer".
- Understanding the goal is rule-based (the phrase set of [moments](moments.md)): it picks out light, weather, actor
  kinds, manoeuvres and whether the goal is about detection quality. Other wording changes nothing in the plan.
- Cost estimates come from job history; with none they are rough, and a study can run longer than estimated — the
  deadline and the GPU budget then cancel it mid-run, and its partial work is lost.
- Failure-rate and search results are read generically from the study's folder; the brief may say only that the
  study finished and where its result is. These two kinds were not run against a simulator in Falcon's tests.
- If the runner is killed outright (`SIGKILL`, a power cut), it cannot cancel its job. Under the printed
  `systemd-run` unit, stopping the unit ends the job with it; otherwise check the Jobs dock. The record shows the
  study as interrupted at the next run.
- GPU-hours count wall time, so a study waiting for the CARLA lock is charged for the wait.
