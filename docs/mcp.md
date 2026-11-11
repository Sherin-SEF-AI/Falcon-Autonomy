# MCP: Falcon in Claude Code

Falcon is also an **MCP server**: Claude Code (or Claude Desktop, or any other client of the Model Context Protocol)
can use Falcon through the same tools Falcon's own assistant uses ([the list](assistant-tools.md)) — read the
project, its runs, jobs and evaluations; edit the project; queue drives, sweeps, trainings and evaluations — in the
Falcon window that is open, or on a project file when none is.

## Setting it up

Register it with Claude Code once:

```
claude mcp add --scope user falcon -- python3 -m falcon_autonomy --mcp
```

With Falcon installed as a command, `claude mcp add --scope user falcon -- falcon --mcp` does the same. From a source
folder, run it with that folder's Python: `claude mcp add --scope user falcon -- python3 -m falcon_autonomy --mcp`
from the folder, or `python3 -m falcon_autonomy.mcp` (the server module itself).

To work on a project file while no Falcon window is open, name it:

```
claude mcp add --scope user falcon -- python3 -m falcon_autonomy --mcp --mcp-project ~/work/city.falcon.json
```

Options: `--mcp-project FILE` (the project when no window is open; `open_project` in a session chooses another),
`--mcp-mode auto|window|file` (auto: the open window, else the file — the default; window: only a window; file: only
the file, even with a window open).

`claude mcp list` shows it; in Claude Code, `/mcp` shows its tools and resources.

## Where the calls go

- **Falcon's window**, when one is open and lets MCP clients in (Settings ▸ Assistant ▸ Provider ▸ *Let MCP clients
  work in this window*, on by default). The window listens on 127.0.0.1 only, on a free port, and writes
  `$XDG_CONFIG_HOME/FalconAutonomy/mcp/<pid>.json` — its port and a random token — in a folder only you can open (0700)
  as a file only you can read (0600); every request carries the token, and a wrong one is refused. With several
  windows open: the one that started the server (`FALCON_MCP_PID`), else the one with `--mcp-project` open, else the
  newest. Every change shows in the window at once, as **one undo step "Claude · <tool>"** in Edit ▸ Undo however many
  edits the tool made; jobs go to the window's queue and show in Jobs.
- **A project file**, when no window is open: the tools run in the MCP server's process as `falcon assistant call`
  runs them. Each act call is one undoable step, **saved as it is made**, and the first time the file is copied to
  `<name>.falcon.json.bak` as it was. A file changed on disk since it was opened (by Falcon's window, an editor, git) is
  never overwritten: the call is refused until `open_project` reads it again; a file open in a window is worked on
  there. Jobs queued on a file run in the MCP server's process and stop with it — open the project in Falcon's window
  for long jobs.

`falcon_status` (and the resource `falcon://status`) says which, with the steps made from here.

## What it offers

- **Tools**: every read and act tool of the assistant, with its full description and schema, annotated (read-only or
  not; none destructive: act tools change things reversibly), validated and privacy-checked exactly as in the app;
  each result whole, as text and as structured content. Then the server's own: `falcon_status`, `undo_last_change`
  and `redo_last_change` (only this client's own steps — in the window, only while the newest step on the undo stack
  is one: a step of yours is left alone), `open_project`.
- **Resources**: `falcon://status`, `falcon://overview` (project_overview), `falcon://runs` (the newest runs),
  `falcon://queue` (system_queue), and the template `falcon://run/{run}` (runs_get) — read through the same tools.
- **Arrival order**: reads run side by side; a call that changes something waits for every call before it and holds
  back the ones after it, so an edit then a read gets the read after the edit.
- **Cancellation**: a cancelled call that has not started never runs; one already running cannot be interrupted (a
  tool runs to its end) but its answer is dropped.
- **Progress**: a call given a progress token gets a progress notification every 2 s while it runs.
- **Logging**: `logging/setLevel` turns on notifications of each call's start, end and errors.
- Protocol versions 2024-11-05, 2025-03-26, 2025-06-18 and 2025-11-25 (the client's when it asks for one of these).

## What is never done from here

- **Moving runs to the trash** — that asks you in Falcon's window every time; MCP clients get a refusal.
- Everything on the assistant's never-list: deleting runs or projects, changing Falcon's settings (ports, server,
  keys), starting or stopping a server outside the job queue, shell commands, pushing or posting, downloading a model
  without your licence approval, and images, video, lidar, audio or anything from your dashcam footage (the tools
  refuse those sources and return text only).
- A step that Falcon asks you to approve on a card in every mode (the research autopilot's compute steps) is refused
  with where to approve it: the card is yours, in the Assistant panel.
- Window tools (the `ui_` group) without a window.

## Limits

- Results go to the MCP client, which sends them to its model provider: the same text-only rules as the in-app
  assistant apply, and Falcon's audit log does not see an MCP client's requests (the client's own log does).
- A tool that is running cannot be stopped (cancellation drops its answer); `jobs_wait` waits up to its own timeout.
- In a file session, a job runs only while the MCP server runs.
- The window is found through its file in your config folder: two users on one machine never see each other's
  windows, and a file left by a window that crashed is removed when its process is gone.

Ported from Vantage's MCP server (`vantage --mcp`), whose protocol layer this shares.
