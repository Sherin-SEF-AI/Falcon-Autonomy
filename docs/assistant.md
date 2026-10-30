# The assistant

Falcon's assistant is an agent that **uses the app through typed tools**: it finds runs, reads evaluations, explains
results with the numbers Falcon measured, edits the project, queues jobs and waits for them — through the same code
as Falcon's own buttons and its command line, and nothing else. It runs on a language-model server with **your own
API key** (Groq by default) and sends **text only**. It is off until you turn it on.

## Setting it up

1. Get an API key at [console.groq.com](https://console.groq.com) (API Keys; a key starts with `gsk_`).
2. In Falcon: **Settings ▸ Assistant** — tick *Use the assistant*, paste the key, press **Apply**. The first time, a
   notice says exactly what is sent and what never is; nothing leaves the machine until you accept it.
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
| **Fast** | `openai/gpt-oss-20b` | one-shot helpers: a summary, *explain this error*, phrasing a suggestion |

The fast lane was planned on `qwen/qwen3-32b`; Groq shut that model down on 2026-07-17 (its deprecations page names
`openai/gpt-oss-120b` as the replacement). The default is the production `openai/gpt-oss-20b` instead; pick any model
the server lists. Falcon still passes the right reasoning parameters if you choose a Qwen 3 model.

Settings ▸ Assistant also sets the temperature (default 0.3), the reasoning effort for models that take it
(gpt-oss: low, medium, high) and the default permission mode.

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

## The panel

- **Chat**: the mode and the main model at the top; the **plan** — the short checklist the model keeps while it
  works; the messages, streamed as Markdown; **tool cards** (collapsed: the tool, how long it took, how it was
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

## What is sent, and what never is

Sent, as text, to the server you configured: your messages and the text files you attach; the conversation so far,
including the tools' results (names, ids, settings and measured numbers of your project, runs, jobs, scenarios, models
and evaluations, passages of the manual); the tools' descriptions and argument schemas; the name of the project and
the page that is open. Before anything leaves the machine every string is redacted: your home folder becomes `~`, and
anything shaped like a credential (a `gsk_…` or `sk-…` key, a bearer token, a `password=`/`token=`/`api_key=` value)
becomes `[secret]`.

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

- Groq's base developer-plan limits (verified 2026-10-30) are 30 requests and 8,000 tokens per minute for the gpt-oss
  models; Falcon's tool schemas and a long conversation can come near that. Your account page shows your limits.
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
- `openai/gpt-oss-120b` (production; 131,072 context, 65,536 max output) and `openai/gpt-oss-20b` take tools but not
  parallel tool use, and `reasoning_effort` `low`/`medium`/`high` (default medium) with `include_reasoning`;
  `reasoning_format` is not supported by gpt-oss. Qwen 3 models take `reasoning_effort` `none`/`default`
  (`qwen/qwen3.8-27b`: low/medium/high) and `reasoning_format` (`parsed` or `hidden` with tools; `raw` with tools is a
  400).
- `qwen/qwen3-32b` is deprecated, shut down on 2026-07-17, replacement `openai/gpt-oss-120b`.
- Errors: 400, 401, 403, 404, 413, 422, 424, 429, 498 (flex capacity), 499, 500, 502, 503, with
  `{"error": {"message", "type"}}`; `retry-after` (seconds) only on 429; `x-ratelimit-*` headers on every response.
- Not documented there, handled defensively: where usage arrives in a stream (Falcon reads a final chunk's `usage`
  or Groq's `x_groq.usage`), and the `tool_use_failed` error code with its `failed_generation`.

Nothing was sent to Groq while building this: the tests run against a fake server (`tests/fake_groq.py`).

## For developers

- `assistant.tool_api` — the tool contract (tiers, the registry, `NEVER`, `call`); `assistant.privacy` — redaction and
  the private-footage guard.
- `assistant.groq.GroqClient` — the HTTP client (standard library); `assistant.provider` — results, errors, model
  parameters, cost; `assistant.keystore` — the key.
- `assistant.agent.Agent` — the loop, Qt-free, driven by a `Host` (the panel's `QtHost`, or a headless one in tests
  and on the command line). Meta-tools it registers: `plan_update` and `plan_wait_jobs`.
- `assistant.service.AssistantService` (`win.assistant`) — what the rest of Falcon calls:
  `quick(prompt, context) -> str` (fast model, blocking: call it from a task), `explain_error(text, where)`,
  `ask(text, context=None)` (opens the panel with a request; sent at once when the assistant is ready),
  `propose({'tool', 'args', 'why'})` (an approval card for a suggested action; approved, it runs like a tool call and
  is noted in the conversation).
- A tool that moves runs to the trash reports each move with `ctx.emit('trashed', {'from': path, 'to': path})` (or
  returns `{'trashed': [...]}`) so the activity log can restore it; project edits and queued jobs are found by the
  agent itself (the undo stack's index and the queue's job ids before and after the call).
