# Team workspace and triage

A project kept on a **shared folder** — an NFS export, an SMB/CIFS share, a NAS — that several people open in their
own Falcon at the same time. There is no server and no account: Falcon's instances coordinate through the files
alone. A team workspace adds who recorded each run, comments on runs and failures, and a **triage queue** of failures
that are assigned, investigated, fixed and verified. Code: `falcon_autonomy/core/team.py` (its docstring is the
reference); the page is **Triage**; the commands are `falcon team`, `falcon triage` and `falcon comment`.

## Making one

Keep the project file and its workspace folder (`NAME.falcon.json` and `NAME.falcon/`) on the shared folder, then
**Triage ▸ Make this a team workspace** (or `falcon team init PROJECT`). That writes `NAME.falcon/team/team.json`;
everything else follows from it. Each member sets **Settings ▸ Team ▸ Your name** (empty: `$FALCON_USER`, else the
login name); the host name goes with it. It is a label, not an account: anyone who can write to the folder can write
as anyone.

What changes in a team workspace:

| | solo workspace | team workspace |
|---|---|---|
| comments, triage | in the workspace's `team/` folder | the same, seen by every member |
| run owner | not recorded | `team/owners/<run>.json`: who started the run, on which machine, when (Data ▸ Browse ▸ Run ▸ *Recorded by*) |
| the catalogue (SQLite) | `NAME.falcon/catalog.sqlite` | a per-user file in `~/.cache/FalconAutonomy/catalogues/`, rebuilt from the run folders when the project is opened |
| saving the project | a plain save | under the project's lock, and refused when another member saved it after you opened it |
| the job queue's CARLA and GPU locks | this machine's (`~/.config/FalconAutonomy`) | the same: they guard this machine's server and GPU, not shared files |

## How it stays consistent without a server

1. **Append-only, one file per fact.** A comment, a triage event, a run's owner and a member's last visit are each a
   small JSON file with a unique name (UTC time to the microsecond, the user, a random suffix), written to a temporary
   name in the same folder and linked (or, where links are refused, created exclusively) into place, never rewritten.
   Two people writing at the same moment write two files; nothing is merged, nothing is lost. Appending lines to one
   shared JSONL is not done: `O_APPEND` is not atomic over NFS.
2. **State is a fold of events.** A triage item is `team/triage/T-<n>/item.json` (written once) and its
   `events/`. Its state, assignee, links and title are the events applied in time order (ties by id). A transition
   that no longer fits the state the earlier events reached — two members marking it fixed at once, one verifying
   what another just reopened — is kept in the history as **ignored**, with why, so an item never ends in a state
   nobody chose. Item numbers are claimed with `mkdir`, atomic on every filesystem here: two members creating items
   at once get two numbers.
3. **Rewritten files are locked.** The few files written whole — the project file, the saved failure clusters
   (`clusters/clusters.json`) — are written under a lock file, and the project only when it has not changed since
   this window read it.

### The lock

`FileLock` creates `FILE.lock` beside what it guards, holding the owner, host, pid, a random token and the time:

- **Creation** links a fully written temporary file onto the lock name (`link(2)`: atomic on local disks and on NFS
  v2/v3/v4 — the method `open(2)` documents for NFS; a lost reply is detected by the temporary file's link count). Where
  hard links are refused (SMB/CIFS, FAT) it falls back to `O_CREAT|O_EXCL`, which SMB servers perform atomically.
- **Heartbeat.** The holder touches the lock file at least every 15 s.
- **Stale locks** — a heartbeat older than 120 s by the file server's own clock (read from a file touched in the same
  folder, so two machines' clocks are never compared), or a holder on this host whose process no longer exists — are
  taken over by renaming the lock aside (atomic: of two takers one wins); a taker that finds it moved a lock refreshed
  meanwhile puts it back. A holder that stalled past 120 s (a suspended laptop) finds its lock lost and leaves the new
  holder's alone.
- `fcntl`/`flock` locks are **not** used: over NFS they need a working lock manager (lockd/NLM, or NFSv4 state) and
  are lost silently with a client's lease; over SMB they become byte-range locks with other semantics.
- `falcon team lock FILE` says who holds a lock and how old its heartbeat is; `--clear-stale` moves a stale one aside.

**Safe on**: local filesystems, NFSv3 and NFSv4 (any mount options; close-to-open consistency is enough), SMB2/3
(Samba, Windows file servers, NAS appliances). **Not safe on** file-sync services that copy files between machines
(Dropbox, Syncthing, OneDrive, Google Drive): there is no shared file to be atomic on, so two machines can each hold
"the" lock. On those, comments and triage still work (rules 1 and 2 need no lock), but two members saving the project
can overwrite each other.

### The catalogue

SQLite's locking is unreliable over NFS and SMB ([sqlite.org FAQ 5](https://www.sqlite.org/faq.html#q5)), so a team
workspace's catalogue is never kept on the share: each member's Falcon keeps its own copy in its cache folder and
rebuilds it from the run folders (the truth) when the project opens (`falcon runs PROJECT --rebuild` does it from the
command line). Data ▸ Browse lists runs from the run index, which reads the folders, so other members' new runs appear
there at once; the counts that read the catalogue (the navigator, Home) catch up on the next open. Links the
catalogue held between a live session and its recordings are not rebuilt.

### Saving the project

The project file is the one shared file people edit. In a team workspace File ▸ Save takes `FILE.falcon.json.lock`,
compares the file with what this window read, and refuses when another member saved it since — naming who, on which
machine and when (`team/last-save.json`). Reopen it and redo your change, or **Save anyway**, which keeps only yours.
Edits are not merged.

## The triage queue

An item is a failure (or any issue) to be dealt with: a title, its links, an assignee, a state, a history, comments.

| state | reached by |
|---|---|
| open | created |
| assigned | assigning it (to anyone, by name) |
| investigating | someone assigned starts on it |
| fixed | the assignee says it is fixed |
| verified | someone checks the fix (the history notes when it was the same person who fixed it) |

A fixed or verified item goes back to **investigating** when the fix does not hold. **Links** are `kind:id`:
`run:runs/<folder>`, `scenario:<id>`, `case:<study>#<row>`, `suite:<id>`, `ci:<name>`, `cluster:<id>`
([failure clusters](failure-clusters.md)), `bisect:<name>` and `commit:<sha>` ([bisect](bisect.md)), `study:<id>`,
`triage:T-<n>`.

**Triage** (the page, under Improve): the queue filtered by state or to your own items; an item's links, history
(changes ignored as concurrent are marked) and comments; *Assign to me*, *Assign…*, *Investigate*, *Fixed*,
*Verified*, *Reopen*. *Triage this cluster* (Failure modes) creates an item linking a cluster and its first runs.
**Comments** appear on triage items and on every run in Data ▸ Browse (one file each, oldest first); a comment the
assistant wrote for you says so.

```sh
falcon team init PROJECT [--name N]                 # make the workspace a team workspace
falcon team info PROJECT [--json]                   # members (last seen), items, who you are
falcon team owner PROJECT RUN                       # who recorded a run
falcon team lock FILE [--clear-stale]               # who holds a shared file's lock
falcon triage list PROJECT [--state S] [--assignee NAME | --mine] [--json]
falcon triage show PROJECT ITEM [--json]
falcon triage add PROJECT "TITLE" [--link KIND:ID]... [--assign NAME] [--kind failure|issue] [--note TEXT]
falcon triage assign PROJECT ITEM [NAME]            # default: you
falcon triage state PROJECT ITEM assigned|investigating|fixed|verified|open [--note TEXT]
falcon triage link PROJECT ITEM [--add KIND:ID]... [--remove KIND:ID]...
falcon comment PROJECT KIND:ID ["TEXT"] [--reply-to ID] [--json]   # without TEXT: list the comments
```

Exit codes: `0` done, `2` refused (a transition that is not a step of the queue, an unknown item, a malformed
reference) or a usage error.

## The assistant

In the assistant's **research** group: `research_triage` reads the queue, one item with its history and comments, or
an object's comments; `research_triage_edit` comments, adds an item, assigns, moves or links one — append-only,
written with your name and `via: assistant` (shown as such). Privacy: an item linking a run from the dashcam footage
is listed by id and state only, and the assistant neither reads nor writes comments on such a run
([assistant](assistant.md)).

## Tests

`tests/test_team.py`: identity; exclusive creation by link and, with links refused, by `O_EXCL`; the lock's holder
named, released, taken over when its heartbeat is old and when its holder's process is gone, kept fresh by its
heartbeat, lost to a takeover without harming the new holder; four processes counting to 60 under it without losing
an update; comments, references and owners; eight triage items created at once by eight threads getting eight
numbers; transitions refused; two "fixed" events written at the same moment folded into one and one ignored; the
project saved over another member's save refused and forced; a team workspace's catalogue on this machine, rebuilt
from the runs; the commands. `tests/test_gui_triage.py`: the page end to end, a run's comments and owner in Data ▸
Browse, the save conflict in the window, Settings ▸ Team. Run on a local filesystem only: the NFS and SMB behaviour
above is from their specifications and the `open(2)` and `link(2)` manuals, not measured on a share here.

## Limits

- **Identity is a label.** No authentication: the folder's permissions are the only access control.
- **Clock order.** Events are ordered by the writing machine's clock; machines whose clocks disagree by more than the
  time between two people's changes can order them unexpectedly (both are kept and shown). Keep NTP on.
- **Not on sync services** for the project file (above).
- **No editing or deleting** of comments and events: a correction is a new comment or a new event.
- **Project edits are not merged**: one member's save wins, and the other is told before it is lost.
- **Other shared files** — a study resumed by two members at once, the run index cache (`runindex.json`, rewritten
  whole, last writer wins; it is only a cache) — are not locked.
