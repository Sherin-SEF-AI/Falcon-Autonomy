# The assistant

Falcon's assistant is an agent that **uses the app through typed tools**: it finds runs, reads evaluations, explains
results with the numbers Falcon measured, edits the project, queues jobs and waits for them — through the same code
as Falcon's own buttons and its command line, and nothing else. It runs on a language-model server with **your own
API key** (Groq by default) and sends **text only**. It is off until you turn it on.

## Setting it up

1. Get an API key at [console.groq.com](https://console.groq.com) (API Keys; a key starts with `gsk_`).
2. In Falcon: **Settings ▸ Assistant** — tick *Use the assistant*, paste the key, press **Apply**. The first time, a
   notice says exactly what is sent and what never is; nothing leaves the machine until you accept it (it is asked
   again when what is sent grows: its version 2 added suggestions and *Explain this error*).
3. **Test connection** asks the server for its model list (one request) and says whether your main and fast models
   are offered. **Refresh** fills the model pickers from that list.
4. Open the panel: **Ctrl+Shift+A**, *View ▸ Assistant panel*, the **Assistant** badge in the status bar, or
   **Ctrl+K ▸ Ask Falcon…** (with the assistant on, whatever you type in the palette can be sent as a request).

The key is stored in the system keyring when the `keyring` package is installed and has a working backend, else in
your secrets file (`$XDG_CONFIG_HOME/FalconAutonomy/secrets.json`, readable only by you). It is never written to
`settings.json`, a project, a run, a report, the job history, the assistant's conversation or audit log, or any log
line; Settings shows where it is stored, never what it is. **Remove** deletes it.

The server is configurable (*Server*): any OpenAI-compatible chat-completions API (`…/v1`), for example a local one.
Parameters a server does not know (`reasoning_effort`, `stream_options`, …) are dropped and the request retried once.

## The two models

| | Default | Does |
|---|---|---|
| **Main** | `openai/gpt-oss-120b` | the agent: plans, calls Falcon's tools, reasons over their results, answers |
| **Fast** | `qwen/qwen3.8-27b` | one-shot helpers: a summary, *explain this error*, phrasing a suggestion |

Either can be any model the server lists — Qwen as the main model too (it takes parallel tool calls; gpt-oss does
not). Groq lists `qwen/qwen3.8-27b` as a **Preview** model, which it may withdraw at short notice (it shut
`qwen/qwen3-32b` down on 2026-07-17 and `qwen/qwen3.6-27b` on 2026-09-14). So the fast lane **falls back to
`openai/gpt-oss-20b`** (production) when the server's model list (Refresh, Test connection) does not have the chosen
fast model, or a call says the model is gone (404) — and says so once, in the panel and the log. Falcon sends each
family its own parameters: gpt-oss gets `reasoning_effort` and `include_reasoning` and `parallel_tool_calls: false`;
Qwen 3 gets `reasoning_format: hidden` (with tools it must be parsed or hidden), `reasoning_effort` (`none` for the fast
lane) and `parallel_tool_calls: true`; a model Falcon does not know gets neither.

Settings ▸ Assistant also sets the temperature (default 0.3), the reasoning effort for models that take it
(gpt-oss: low, medium, high), the default permission mode and your plan's tokens-per-minute limit.

## Permission modes

Every tool has a tier: **read** (looks, changes nothing), **act** (changes something reversibly: a project edit, a job
queued, a file written into the workspace, the window navigated) or **trash** (moves runs into the workspace's trash
folder).

| Mode | Read | Act | Trash |
|---|---|---|---|
| **Ask** (default) | runs | waits for your approval card | asks |
| **Auto** | runs | runs | asks |
| **Autopilot** | runs | runs | asks |

An **approval card** shows the tool, its arguments, the exact change (the tool's preview: the command, or the edit as
a diff), its estimate (time, what it does), and **Approve**, **Approve all of this kind** (for the rest of this
conversation; not offered for trash) and **Reject**. A rejection goes back to the model as the user declining.

**Autopilot** works through a goal across many steps and long jobs: while its plan has open items it is told to go on
(until it answers without calling a tool after being told — it is done, or blocked, and says which); when jobs it
queued are still running it waits for the job queue to say one finished (a subscription, not polling) and continues
with what happened. There is no step or token budget: **Stop** (the button, or Esc in the message box) always ends
it — it closes the request in flight, answers a waiting approval card with *stopped* and ends a wait for a job. In any
mode the model can ask to be resumed when a job it queued finishes (`plan_wait_jobs`).

## Tools on demand, and the per-minute token limit

Falcon's 99 tools ([the list](assistant-tools.md)) are in eight groups whose compact schemas weigh, in tokens
(characters / 4), about: project 3,000 (25 tools), jobs 3,200 (29), analysis 940 (11), lab 860 (7: hardware in the
loop, deployment, CI mode, log replay, adversarial traffic, experiments, VLM evaluation), runs 850 (11), ui 650 (8),
knowledge 310 (4), system 180 (4) — 9,960 in all, more than Groq's base per-minute limit. So each step sends:

- the **core**: the runtime's own `plan_update`, `plan_wait_jobs`, `tools_find` and `tools_load`, with the two small
  groups knowledge and system (about 1,200 tokens together);
- the **single tools** loaded: `tools_find` searches every tool's name and full description and makes its best three
  matches callable on their own (so the model can call `jobs_drive` without the other 28 jobs tools); `tools_load`
  loads more by name; a tool the model calls by name before it was loaded is run, and loaded on its own;
- the **groups** loaded — by `tools_load`, or with your message when its words (or the page you are on) make a group
  obviously relevant: "runs", "recorded" load runs; "evaluation", "mAP" analysis; Data loads runs, Models analysis (a
  heuristic, at most two groups per message). The most recently loaded group is always sent; older ones only while the
  step stays within your per-minute limit — so at 8,000 tokens a minute one large group (project or jobs) goes at a
  time; a larger limit sends more.

What is loaded is kept with the conversation.

Groq's rate-limits page gives the base developer plan 30 requests, 8,000 tokens per minute and 200,000 per day for
`openai/gpt-oss-120b`, `openai/gpt-oss-20b` and `qwen/qwen3.8-27b` (its models page says 250,000 tokens per minute;
**your account's limits page at console.groq.com is authoritative**). Settings ▸ Assistant ▸ *Token limit* is that
number for your plan (default 8,000; 0: no limit). Before each step Falcon estimates its tokens (characters / 4):

- a step larger than the limit is trimmed — older tool results left out first (the last four kept), then every tool
  not called in the last two steps, then all but the last result and the core alone; one that still does not fit is
  not sent, and you are told why;
- a step that fits, but not on top of the tokens the last minute's responses used (as the server counted them), waits
  until enough of them age out (the panel says how long; Stop ends the wait);
- a 413 from the server is trimmed the same way and sent again, smaller; a 429 waits for its `retry-after`.

The status bar badge's tooltip shows the last minute's tokens against the limit.

## The panel

- **Chat**: the mode, the main model and **Show me** ([below](#show-me-mode)) at the top; the **plan** — the short
  checklist the model keeps while it works; the messages, streamed as Markdown; **tool cards** (collapsed: the tool, how long it took, how it was
  approved, what it returned; expanded: the arguments and the result as sent to the model); **approval cards**; the
  **jobs it queued** with their progress (*Jobs* opens the Jobs panel on that row); Falcon's own events (a job
  finished, Autopilot going on). Run names, job numbers of this session and scenario ids are links to where they are
  shown. Enter sends, Shift+Enter starts a new line, Esc stops; the file button attaches a text file.
- **Activity**: everything the assistant changed, newest first, each with **Undo** — project edits through the
  window's undo stack (only while they are the newest steps on it; else undo the later ones first), a queued job by
  cancelling it, runs moved to the trash by moving them back.
- **⋯**: *Show what will be sent before each message* (the exact JSON of the messages after redaction, and the tools
  by name, with Send and Cancel), *Clear the conversation*, *Open the audit log*, *Settings ▸ Assistant*.
- The **status bar** badge: the model, the tokens of this session and their estimated cost.

The panel docks at the right when the window has room beside the pages, else it floats over the window's right edge
(a 1280 px screen): showing it never makes the window wider.

Each project has its own conversation, saved in its workspace (`WORKSPACE/assistant/conversation.json`) after every
step and resumed after a restart; *Clear* empties it. It holds the messages, the plan, the tools you approved for the
whole conversation, the activity log and the tokens used — never the key.

## Suggestions

Falcon watches the project for next steps worth taking and shows them as one-click **chips** under the heading of the
page they concern (Worlds, Scenarios, Live, Data, Rare cases, Training, Stack testing; at most three a page) and the
best of all as the **Next best action** on Home (the work's own *Next step* when there is none). Each is found by a
fixed rule from the app's state and the workspace's files (`assistant.suggest`), carries the exact tool call that acts
on it, and says why with numbers read from the files:

| Trigger | When | The call |
|---|---|---|
| server crashed | Falcon's CARLA server is in the crashed state | `system_server` (its state and log tail) |
| job failed | a job of this session failed (its exit and last error) | `jobs_get` with 30 log lines |
| avoidable failures | the newest study with failures at its threshold that RSS labels *avoidable* | `jobs_replay_failure` of the worst |
| failed MRM checks | a stack suite's runs failed the minimal-risk-manoeuvre check (the check failing most) | `jobs_why` on the worst run |
| GPU idle | CARLA jobs wait, none runs, and the GPU is under 10 % | `system_queue` (who holds the CARLA lock) |
| traffic lights | an imported world's signal records control no lane, or a left-hand world was converted before the left-hand fix | `jobs_apply_map_edits` (a re-import: a new world) |
| coverage gap | the recorded drives leave world × time of day × weather cells empty (from the run index, no image read) | `jobs_drive` with the preset that fills the first |
| near-duplicates | the last *Data ▸ Health* report found 20 % or more near-duplicate frames | `runs_tag` them `near-duplicates` |
| drift refusals | a translated run kept raw frames its drift check refused | `ui_show_run` |
| job finished | a job of this session finished in the last 15 minutes | `ui_open_job` |
| first drive | worlds and rigs, nothing recorded | `jobs_drive` (10 s) |
| no evaluation | models and drives, no evaluation | `jobs_evaluate` on the newest drives |
| no scenarios | no scenario of the project's own | `project_scenario_create` (a copy of lead_brake) |
| no rules | no trigger rules | `project_trigger_add` (an example rule, checked when added) |

**A click** opens the Assistant panel with that call, as your permission mode says: in Ask an approval card with the
call's exact preview and estimate (nothing runs until you approve); in Auto and Autopilot a change runs at once (and is
in the activity log with Undo); a read tool runs in every mode; moving runs to the trash always asks. It is noted in
the conversation, so the assistant knows what you did. **×** (or *Not now* on Home) dismisses a suggestion for this
project — kept in `WORKSPACE/assistant/suggestions.json`, so it does not come back after a restart.

**Wording.** Without a key or the notice accepted, suggestions keep Falcon's own wording: they always work offline.
With the assistant on, the **fast model** ranks the top four and words each in one short sentence; its wording is
kept only when every number in it is one of that suggestion's own facts (the grounding check), else Falcon's is
shown; Home says when a wording is the assistant's. What it is sent: the candidates' titles and their one-line facts
(names, ids and numbers of your runs, studies, suites, worlds and jobs), the project's name and the page — never a
private item: runs, studies and worlds from your dashcam footage never become suggestions.

**Never noisy.** The rules run at most every 30 s (a job ending or the project being replaced brings the next run
forward, at least 3 s apart), on a background thread; the model is asked at most every two minutes and only about
suggestions it has not worded yet (a suggestion whose numbers changed is new). Settings ▸ Assistant ▸ *Suggestions*
turns them all off.

## Ask about this and Explain this error

**Ask about this** is in the context menu of the runs in *Data ▸ Browse* (the run search's results) and on Home, the
failing cases in *Rare cases*, the results of a stack suite, an evaluation's scores in *Models*, a job's row in the
Jobs panel and its log window. It opens the panel with a request about that one object, named by its id — a run by
`runs/<folder>`, a job by its number, a case by study and row, a suite result by suite, scenario and seed, an
evaluation by its folder — and the assistant reads it through its tools. Nothing is pasted: no image, no log text, no
path outside the workspace; a private object (anything from the dashcam footage) is refused before anything is asked.
When the assistant is not set up, the request waits in the message box (with what it is about) until you send it.

**Explain this error** is the *Explain* button of a failed job's row in the Jobs panel, its context menu, and a
failed job's log window (in *This session* and in *History*). A card in the panel says why it most likely failed and
how to fix it. With the assistant on, the **fast model** gets the job's title and kind, what its exit code means and
the last 30 lines of its log (redacted: your home folder shortened to `~`, credentials replaced), with the manual's
sections that match the error (searched offline); numbers in its answer that are not in what it was given are marked.
Without a key — or for a job that worked on private data, whose log is never read — the card is worded from the exit
and the last error line alone. When the log matches a failure with an obvious next step, the card has a button for it
(an approval card as above): the server not answering → `system_server`; refused for memory → `system_machine`; the
GPU taken → `system_queue`; a rig without a microphone array → `project_sensor_add` of an `audio_array`; CARLA's Python
missing → what that setting does; weights not fetched → the manual's section (fetching weights and accepting their
licence is yours). The model chooses among these or none; it never starts a server or changes a setting.

## Show-me mode

**Show me** (the toggle at the top of the panel; kept as a setting) asks the assistant to point at what it explains:
before each step it writes one short sentence of what it is showing, then opens the page and tab (`ui_open_page`),
outlines the control you would use (`ui_controls`, `ui_highlight`), or shows the run or frame it talks about
(`ui_show_run`, `ui_show_frame`) — one step at a time, with a pause after each (1.6 s; a highlight's own time, at most
3 s) so you can follow; Stop ends a pause. A *How do I…?*, *Where is…?* or *Show me…* question is answered this way.
With show-me on, the window group is loaded with your message, and moving the view runs without an approval card
(the cards say *show me*); everything else asks as your permission mode says, and the assistant clicks nothing for
you.

## What is sent, and what never is

Sent, as text, to the server you configured: your messages and the text files you attach; the conversation so far,
including the tools' results (names, ids, settings and measured numbers of your project, runs, jobs, scenarios, models
and evaluations, passages of the manual); the tools' descriptions and argument schemas; the name of the project and
the page that is open; with suggestions on, the candidate next steps and their facts (to be ranked and worded); when
you press *Explain*, a failed job's title, exit and log tail with passages of the manual. Before anything leaves the
machine every string is redacted: your home folder becomes `~`, and anything shaped like a credential (a `gsk_…` or
`sk-…` key, a bearer token, a `password=`/`token=`/`api_key=` value) becomes `[secret]`.

Never sent: images, video, lidar, radar, audio, or anything from your dashcam footage. The tools refuse those sources
(`assistant.privacy`: anything under `~/Videos`, and any run, reconstruction, image set, translator or world whose own
records say it came from dashcam footage), and Falcon refuses a message that names such a path and an attachment that
is one, or that is not plain UTF-8 text (images, video, audio, point clouds, archives, model weights), or that is over
256 KB. Your key goes only to the server, in the request's header.

## What the assistant never does

No tool exists for these, in any mode, and a tool by such a name is refused even if one were registered
(`tool_api.NEVER`):

- delete runs or projects — runs can only be **moved to the workspace trash**, with your approval each time;
- change Falcon's settings: its ports, its server, the key;
- start or stop a server outside the job queue, or touch a CARLA server you started;
- run a shell command;
- push, post or publish anywhere;
- download a model without you accepting its licence;
- send images, video, lidar, audio or anything derived from your dashcam footage to the model.

## The audit log

Every request to the server — a chat step, a fast-lane answer, a model list, a connection test — is one line in
`WORKSPACE/assistant/audit.jsonl`: the time (UTC), what it was for, the model, the server's host, the HTTP status, the
attempts, the token counts, the tools the answer called and the **SHA-256 of the exact payload sent** (not the
payload). ⋯ ▸ *Open the audit log* shows it.

## Grounding

After every answer Falcon checks its numbers against the conversation (`assistant.grounding`): each number must be
one a tool returned, an argument of a tool call Falcon ran, a number you wrote or one of Falcon's job events —
rounded to the precision shown (0.734 → "0.73"), as a percentage of a fraction ("73.4 %"), with a k/M/B suffix
("1.2k"), or the length of a list a tool returned ("5 runs"). The model's own earlier answers do not count. A number
that is none of these is marked in the answer and listed under it as **not from Falcon's data**.

Limits: not checked are numbers in code spans and code blocks, numbers that are part of a name (a run id,
`gpt-oss-120b`, ResNet50, 3D), versions, dates and times, list markers, and 0, 1 and 2 standing alone. A difference or
ratio the model computed from Falcon's numbers is marked (it is not a measurement Falcon made). The check finds
invented measurements; it does not prove an answer right.

## Costs

Every response's token counts come from the server. The cost shown is an **estimate from your own price table**
(Settings ▸ Assistant ▸ *Prices*: per model, $ per million input and output tokens) — your configuration, not a claim
of what you are billed. The defaults are what Groq's model page listed on 2026-10-30: `openai/gpt-oss-120b` 0.15 /
0.60, `openai/gpt-oss-20b` 0.075 / 0.30, `qwen/qwen3.8-27b` 0.80 / 4.00. A model without a price shows tokens only.

## Errors

Every failure is a sentence, never a stack trace and never the key: a refused key (401), a model the server no longer
offers (404 — choose another; Refresh lists what is offered), a conversation too long for the model's context (Falcon
first leaves out older tool results and retries; then asks you to clear), a request over your plan's tokens-per-minute
limit (413), a model that does not take tools, the network down. Rate limits (429) and server errors (498, 5xx) are
retried with exponential backoff and full jitter, or after exactly the `retry-after` the server gives (plus up to
10 %), up to four times; the panel says it is waiting. A tool call the model wrote that is not valid JSON goes back to
it as an error to correct; one the server itself could not parse (`tool_use_failed`) is asked again once.

## Limits

- Suggestions are rules, not judgement: each looks at one kind of evidence (the newest study, the newest suite, the
  last health report this session) and suggests one call; the model only ranks and words them. The coverage rule
  counts frames from the run index by world, time of day and weather kind — not labels, not object classes.
- Explain this error reads the log the job wrote; a job that failed before logging anything is explained from its
  exit code alone, and a known fix is matched by words in the log (a heuristic).
- The per-minute limit is paced from an estimate (characters / 4) and the server's own counts of the last minute's
  responses; the server's count can differ, and a 429 then waits for its `retry-after`. Requests per minute and per
  day are not paced (a 429 says when they are reached).
- Tools on demand rest on the model loading what it needs: a model that does not call `tools_find` or `tools_load`
  answers with the core tools and the groups the user's words loaded.
- Context: about 70 % of the model's context window (from the server's model list; 131,072 tokens for gpt-oss) is
  used before older tool results are left out; each tool result is cut at 16,000 characters.
- Stop ends the turn at once, but a tool already running finishes on its own (its result is dropped; what it changed
  is in the activity log only if it finished before Stop).
- Read tools run side by side off the GUI thread; act and trash tools run one at a time on the GUI thread (they edit
  the project through its undo stack and queue jobs), so a slow act tool would hold the window — tools that wait are
  read tools.

## What was verified (2026-10-30)

From Groq's documentation (console.groq.com/docs: models, deprecations, reasoning, tool use, rate limits, errors,
API reference), read on 2026-10-30:

- The chat-completions API at `https://api.groq.com/openai/v1` takes `tools`, `tool_choice` (`none`, `auto`,
  `required`, or a named function), `parallel_tool_calls` (default true), `stream` (server-sent events ending with
  `data: [DONE]`), `stream_options`, `temperature` (0–2), `max_completion_tokens`; `GET /models` lists each model's
  `id`, `active`, `context_window`, `owned_by`.
- `openai/gpt-oss-120b` (production; 131,072 context, 65,536 max output; $0.15 / $0.60 per million tokens) and
  `openai/gpt-oss-20b` (production; $0.075 / $0.30) take tools but **not parallel tool use**, and `reasoning_effort`
  `low`/`medium`/`high` (default medium) with `include_reasoning`; `reasoning_format` is not supported by gpt-oss.
- `qwen/qwen3.8-27b` (**Preview** — "may be discontinued at short notice"; 131,072 context, 16,384 max completion;
  $0.80 / $4.00): tool use and **parallel tool use**, JSON mode; `reasoning_effort` `none`/`default`/`low`/`medium`/
  `high` (`none` and `default` only for this model; `default` returns no reasoning tokens); `reasoning_format`
  `parsed`/`raw`/`hidden`, which must be `parsed` or `hidden` with tools or JSON mode (else a 400).
- Deprecations: `qwen/qwen3-32b` shut down on 2026-07-17 (replacement `openai/gpt-oss-120b`); `qwen/qwen3.6-27b` shut
  down on 2026-09-14 (replacement `qwen/qwen3.8-27b`).
- Rate limits, base developer plan: 30 requests, 8,000 tokens per minute and 200,000 tokens per day for
  `openai/gpt-oss-120b`, `openai/gpt-oss-20b` and `qwen/qwen3.8-27b` (the models page says 250,000 tokens per minute;
  the account's own limits page is authoritative).
- Errors: 400, 401, 403, 404, 413, 422, 424, 429, 498 (flex capacity), 499, 500, 502, 503, with
  `{"error": {"message", "type"}}`; `retry-after` (seconds) only on 429; `x-ratelimit-*` headers on every response.
- Not documented there, handled defensively: where usage arrives in a stream (Falcon reads a final chunk's `usage`
  or Groq's `x_groq.usage`), and the `tool_use_failed` error code with its `failed_generation`.

Nothing was sent to Groq while building this: the tests run against a fake server (`tests/fake_groq.py`).

## Evaluating models

`falcon assistant eval` measures how well a model does Falcon's work through the assistant: about twenty realistic
requests (`falcon_autonomy/assistant/eval_tasks.json`) — *find my worst rare case in the lead_brake study and replay
it*, *record a 10 s night drive in the Bengaluru world and check its dataset health*, *job N failed: why?*, *add a
trigger rule that marks close calls*, *compare my last two drives*, *tag every rainy run*, *move this run to the
trash* (the harness answers the approval card with no), a request about a run from the dashcam footage (it must be
refused), a request to change a setting (it must not act) and more — each run end to end by the real agent with its
permission mode, its tools on demand and your per-minute token limit.

```
falcon assistant eval --out eval-results                     # your main model (Settings ▸ Assistant)
falcon assistant eval --model openai/gpt-oss-120b --model qwen/qwen3.8-27b --out eval-results
falcon assistant eval --tasks my-tasks.json --only explain_failed_job,add_trigger_rule --out eval-results
```

It uses your key, your server, your prices and your token limit as Settings ▸ Assistant has them (read, never
changed), and is refused — exit 2, nothing sent — until the assistant is on, its notice accepted and a key stored. It
does send requests: each task is several model requests, each carrying the conversation and the tool schemas sent
that step (see *Tools on demand*), so a run takes a real share of a plan's daily tokens; your per-minute limit paces
them, and `--only` runs a few tasks.

**Nothing touches your work.** Each task runs on its own throw-away project (the default world and rig with a small
camera) in a temporary folder that is deleted afterwards; its jobs run on the fake carla (`tests/fake_carla` in
Falcon's source folder: the workers run against a stand-in for CARLA's API, no server is started) on a job queue with
its own locks — your workspace, your settings and any CARLA server are never used. The *Bengaluru* world of that task
is a Town10HD stand-in the task's setup adds under that name: the task measures whether the model finds and uses the
right world, not an OSM import.

**How success is checked.** A task's setup is declarative and goes through the tools themselves (drives queued and
waited for, a small sweep, a drive whose server never answers, a run marked as dashcam-derived — a marker, no footage).
Its criteria are machine-checked after the turn (and after the jobs the assistant queued have ended): the tool calls
made (with the arguments that matter), calls and tiers that must not be used, a refusal returned by a tool, how
approval cards were answered, the project afterwards (a trigger rule matching `actor.ttc`, 20 vehicles in the default
profile), runs still there, read tools called afterwards (`runs_search tag:rain` finds one run), the jobs' outcomes,
words the answer must contain, and no number flagged by the grounding check. A task fails with the criteria it missed,
in words. A task file of your own uses the same format (`assistant/evaluate.py`, `TASK_SCHEMA`).

**What it reports**, per model, in `DIR/summary.md` (printed) and `DIR/results.json`: the success rate, the mean
number of model requests per task, the tokens (the server's counts), the mean wall time of a turn and the time spent
waiting for the model, the **estimated** cost from your price table (blank for a model without a price — it is your
configuration, not a bill), and the tasks whose answers had numbers not from Falcon's data; then each task with why it
failed. `DIR/audit/` keeps each task's audit log (no payloads). Exit 0 when every task passed, 1 otherwise.

No results are given here: none were measured with a real key while building this (the tests run the tasks against
the fake server with scripted answers). Run it yourself; results depend on the model, the server's version of it and
the day.

## For developers

- `assistant.tool_api` — the tool contract (tiers, the registry, `NEVER`, `call`); `assistant.privacy` — redaction and
  the private-footage guard.
- `assistant.groq.GroqClient` — the HTTP client (standard library); `assistant.provider` — results, errors, model
  parameters, cost; `assistant.keystore` — the key.
- `assistant.agent.Agent` — the loop, Qt-free, driven by a `Host` (the panel's `QtHost`, or a headless one in tests
  and on the command line). It imports `assistant.tools` (which registers Falcon's tools in `tool_api.REGISTRY`). Its
  own meta-tools (`agent.CORE`: `plan_update`, `plan_wait_jobs`, `tools_find`, `tools_load`) are kept out of the
  registry and run by `agent.call_tool` with the same checks; `agent.ALWAYS` (knowledge, system) is sent with them;
  everything else once loaded, singly or by group. `tools_find` searches names and the full texts
  (`tools._compact.DETAILS`) and shows the compact first sentence.
- `assistant.service.AssistantService` (`win.assistant`) — what the rest of Falcon calls:
  `quick(prompt, context) -> str` (fast model, blocking: call it from a task), `explain_error(text, where)`,
  `ask(text, context=None)` (opens the panel with a request; sent at once when the assistant is ready),
  `propose({'tool', 'args', 'why'})` (a suggested action: an approval card in Ask — or at once when the permission
  mode lets it; it runs like a tool call and is noted in the conversation).
- `assistant.suggest` — the suggestion triggers (`TRIGGERS`, each `fn(State) -> [Suggestion]`), `phrase()` (the fast
  model's ranking and wording, grounding-checked), `Store` (dismissals) and `Pacer`; `ui.suggestions` — the engine
  (`win.suggestions`), the chips and Home's card. `assistant.explain` — `job_context`, `known_fixes`, `manual_for`,
  `explain`/`offline` (Explain this error) and `about()` (Ask about this: ids only, `Refused` for private objects);
  the window's `explain_job(job)` and `ask_about(kind, **ids)`.
- Show-me: `Agent.show_me` (the panel's toggle) adds `agent.SHOW_ME` to the system prompt, loads the ui group, runs ui
  act calls without a card (approval `show_me`) and pauses after each (`SHOW_ME_PAUSE_S`, the host's `show_me`
  event).
- `runs_trash` returns `{'trashed': [{'from', 'to', 'item'}]}` (and emits each move as `ctx.emit('trashed', …)`);
  the activity log keeps the item and its Undo calls `runs_restore` with it, which puts the catalogue back too.
  Project edits and queued jobs are found by the agent itself (the undo stack's index and the queue's job ids before
  and after the call). Act and trash tools must be quick (they run on the GUI thread); `jobs_wait` is a read tool
  and waits off it (a Stop leaves it waiting until its own timeout, at most 600 s, its result dropped).
