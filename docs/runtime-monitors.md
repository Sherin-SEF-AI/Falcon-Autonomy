# Runtime monitors generated from the rules

Falcon checks [traffic rules](traffic-rules.md) and [STL requirements](falsification.md) offline, on recorded runs.
A **runtime monitor** is the same formula compiled into an online evaluator that runs on the vehicle computer (a
Jetson AGX Orin with JetPack 6 and ROS 2 Humble is what the package is written for), so the rule that failed a run in
simulation is the rule watching the car. Code: `falcon_autonomy/core/monitorgen.py` (compile, generate, check,
deploy, the command), `core/monitor_runtime.py` (the evaluator: standard library only; copied verbatim into the
package), `core/monitor_node.py` (the rclpy node, copied verbatim); their docstrings are the reference.

## Where it shows

| where | what |
|---|---|
| Hardware ▸ *Runtime monitors* | a rule set, *Generate*; the package's rules with their equivalence status in a word and a colour — equal, differs, not checked, formula changed — the verdict delay and the formula hash; *Check against a run…* (the check wears the fidelity of the runs it replayed: `Recorded · of <level> runs`); the manifest's id short, click to copy; three commands, each with a **Copy** button — the dry run, the deploy and the rollback — and, behind *On the device*, the install commands. Falcon runs none of them |
| `falcon monitor gen (--rules india\|generic\|FILE [--rule ID …] \| --stl FORMULA) (--out DIR \| --project P) [--name N] [--rate HZ]` | writes the package |
| `falcon monitor check RUN… --monitor DIR [--xodr FILE] [--json]` | replays the runs through the generated module and compares with the offline checker; writes `DIR/equivalence.json`; exit 0 agree, 1 disagree, 2 failed |
| `falcon monitor deploy DIR [--dry-run]` | copies the package to the device over SSH (the HIL device of your settings) and loads it once there, after keeping the version it replaces; prints the `systemctl` commands and runs none of them. `--dry-run` validates the package **here** (no SSH, no network) and prints exactly what would be copied and enabled; exit 2 when it is not deployable |
| `falcon monitor rollback DIR [--dry-run]` | restores the version the last deploy replaced on the device and restarts its user unit if it is running (4.2); `--dry-run` prints what would be run there |
| `falcon monitor bench (DIR \| --rules SET) [--steps N] [--json]` | the per-step cost on the machine it runs on |
| `falcon monitor show DIR` · `falcon monitor list --project P` | a package's rules, hashes and status |
| the assistant | `formal_monitor_gen`, `formal_monitor_check` ([assistant tools](assistant-tools.md)) |
| [the safety case](gsn.md) | a Solution binds a monitor as evidence (`--kind monitor --ref monitors/<name>`) |

## What is monitored, and when a verdict is known

A rule of a rule set must hold **at every sample** (`core.rules`: G φ). The monitor evaluates φ at each sample with
`core.stl`'s quantitative semantics — robustness ρ (ρ ≥ 0 satisfied, in the predicate's units), `and` = min, `or` =
max, windows `[t+a, t+b]` and `[t−b, t−a]` with the same 1e-9 s tolerance, and the same three-valued treatment of
unknown samples (NaN: an unknown never makes a violation). A free formula (`--stl`) is monitored the same way, at
every sample; an outer unbounded `always` (`G φ`) is taken as that "at every sample" and removed.

Each step returns, per rule, a **verdict** — `ok`, `violated`, `unknown` (the robustness of the decided samples is
unknown: a signal is missing or stale) or `pending` (nothing could be decided at this step) — the **robustness** of
the samples decided at this step (the lowest), the samples decided (index, time, ρ), how many are still pending, the
worst robustness so far and the number of violated samples so far.

What differs from the offline checker is only *when* a sample's value is known:

| the formula | online |
|---|---|
| no temporal operator, or only past ones (`once`, `historically`, `since`) | decided at its own sample: no delay |
| a bounded future operator (`eventually[a,b]`, `always[a,b]`, `until[a,b]`) | sample *t* is decided when a sample later than *t* + the formula's future horizon arrives; `pending` until then |
| an unbounded future operator inside the formula (`eventually[a,inf]`, a nested `always`, `until` without a bound) | **refused** at generation, with the operator named: its value is never known online |
| `since` without a bound | **refused** (this evaluator keeps since's window; `once` and `historically` may be unbounded — a running maximum / minimum, constant memory) |
| a formula that reads `pet_s` or `accel_mps2` | **refused**: those signals are non-causal offline (`pet_s` reads the whole run; `accel_mps2` is a central difference) |

The built-in rule sets (`india`, `generic`) use only `once[0,c]` and `eventually[0,c]` and are monitorable in full.
One rule looks ahead: `keep_left` / `keep_right` allow driving out of the kerb lane within `overtake_window_s` (10 s)
*before* a pass, so its verdict for time *t* is known at *t* + 10 s. Every other rule is decided at its own sample.

At the end of a recorded trace, `finish()` decides the samples still pending with their windows cut at the end —
`core.stl`'s finite-trace rule. A monitor on a vehicle never ends, so its last horizon simply stays pending.

Memory is bounded by the windows: per temporal operator, the samples inside its window plus those waiting on the
future horizon (about window length ÷ sample period), with a hard cap per operator (`max_buffer`, 200 000 samples)
that raises an error instead of growing. Sample times must increase by more than 1e-6 s.

## The package

`falcon monitor gen` writes a self-contained folder; nothing in it imports `falcon_autonomy`:

| file | what |
|---|---|
| `falcon_monitor.py` | the evaluator (`Monitor`, `MonitorSet`, `SignalMapper`): Python standard library only |
| `rules.json` | the compiled rules as data: per rule its formula in normal form, its tree, signals, horizon and hash |
| `falcon_monitor_node.py` | the ROS 2 node (rclpy): subscribes to the mapped topics, samples them at a fixed rate (default 20 Hz), steps the monitors, publishes the outputs |
| `mapping.json` | topics → signals. **You edit this.** It is kept when the package is generated again |
| `falcon-monitor-<name>.service` | a systemd *user* unit (no root needed) |
| `README.md`, `manifest.json` | how to run it; every file's SHA-256 (not `mapping.json`'s) and each rule's formula hash |

**The mapping.** The rule signals (`speed_mps`, `lead_gap_m`, `lane_margin_m`, … — `falcon rules signals` lists
them) are numbers per sample. The default mapping expects each on `/falcon/signals/<name>` as
`std_msgs/msg/Float64`; `speed_kmh` is derived from `speed_mps` (× 3.6); `t` is the node's own clock, in seconds from
its first tick. An entry is `{topic, type, field, scale, offset}` (a message field by a dotted path, e.g.
`{"topic": "/odom", "type": "nav_msgs/msg/Odometry", "field": "twist.twist.linear.x"}`), `{from, scale, offset}`
(derived from another signal) or `{const}`. A signal that is not mapped, was never received, or is older than
`stale_s` (0.5 s) is *unknown* at that sample.

**The outputs.** `/falcon/monitor/alarm` (`std_msgs/msg/Bool`: a rule is violated at this step),
`/falcon/monitor/verdict` (`std_msgs/msg/String`, JSON: per rule verdict, robustness, worst, violations, pending) and
`/diagnostics` (`diagnostic_msgs/msg/DiagnosticArray`: per rule OK / WARN for unknown and pending / ERROR). When the
clock jumps back (a simulation restarted) the monitors and the received values are reset.

**Hashes.** A rule's `sha` is `stl-` + the first 16 hex digits of the SHA-256 of `{formula, signals_version}` in
canonical JSON, the formula in `core.stl`'s normal form with the rule set's constants substituted — the same
construction as the falsifier's `spec_key` (`core.stlobjective`), without its scale and smoothing, which do not
change what is monitored. The falsifier's own key of the same requirement is recorded beside it (`spec_key`), so a
monitor and a falsification study of one requirement can be joined.

## Equivalence: the generated monitor against the offline checker

`falcon monitor check RUN --monitor DIR` loads **the package's own `falcon_monitor.py`** (after checking the files
against the manifest's hashes: a changed file is refused), replays the run's signals (`core.rules.signals` of its
trace) through it one sample at a time, then `finish()`, and requires:

- every sample's robustness equal to `core.stl.evaluate`'s on the whole run — the same floats, unknown where unknown
  (what the monitor decided *before* the end is therefore already the final value);
- the run's verdict (violated / satisfied / not evaluable) and lowest robustness equal to the offline ones, and to
  `core.rules.check`'s verdict when the monitored rules are still the rule set's.

The result — agreement per rule, and for a disagreement the first sample with its time and both values — goes to
`DIR/equivalence.json`, with the manifest hash it was made on. The tests make the same comparison on 500 random
formulas (nested bounded future and past operators, until and since, unbounded once / historically, unknown and
infinite samples, non-uniform steps) and on both built-in rule sets over random signals, and on a run recorded on
the fake CARLA.

## The safety case

`falcon gsn bind NODE --kind monitor --ref monitors/<name>` binds a package's manifest to a Solution (its SHA-256
pinned, as for any evidence). With no claim, verification reads the package:

| status | when |
|---|---|
| green | the latest equivalence check of this manifest agreed for every rule |
| amber | a monitored rule's formula has **changed since the monitor was generated** (the rule file was edited: the car is no longer watched by the rule that is checked offline) or the rule is gone; no equivalence check on this manifest yet; the package's files changed; or the manifest changed since it was bound (generated again: re-bind) |
| red | the latest check found a disagreement (the first sample is in the reason) |

## Deployment

`falcon monitor deploy DIR` reuses the HIL device settings and `hil/deploy.py`'s SSH (keys only, BatchMode, nothing
waits for a prompt): it makes `~/falcon-monitor/<name>` on the device, copies the package's files, and loads the
monitor once with the device's Python (no ROS needed for that). It installs, enables and starts **nothing**, and never
runs `sudo`: it prints what you run on the device —

    mkdir -p ~/.config/systemd/user && cp ~/falcon-monitor/<name>/falcon-monitor-<name>.service ~/.config/systemd/user/
    systemctl --user daemon-reload
    systemctl --user enable --now falcon-monitor-<name>.service
    sudo loginctl enable-linger "$USER"     # only to keep it running without a login session; yours to run

Before it copies, deploy keeps what is already on the device: if `~/falcon-monitor/<name>` exists it is copied to
`~/falcon-monitor/.previous/<name>` (the copy replaces an older one there). Only one previous version is kept.

### Dry run (4.2)

`falcon monitor deploy DIR --dry-run` reaches no device and opens no connection. It checks, on this machine:

| check | passes when |
|---|---|
| manifest | `falcon.monitor/1`, a valid name, and its `hash` is the hash of its own contents |
| pinned files | every file the manifest pins is present with the manifest's SHA-256 |
| mapping.json | the evaluator's own `SignalMapper` reads it (it is yours to edit: not pinned) |
| rules.json | the package's own `falcon_monitor.py` loads it with as many rules as the manifest lists |
| systemd unit | the unit is byte-for-byte the one `gen` writes for this name (a user unit, `WorkingDirectory` and `ExecStart` in `%h/falcon-monitor/<name>`, no root) |

then prints the folder the current version would be kept in, every file it would copy with its size and the first 16
hex digits of its SHA-256, and the commands that would enable the unit — and exits 2 (sending nothing) when a check
fails.

### Rollback (4.2)

`falcon monitor rollback DIR` (DIR names the package, hence its name on the device) swaps `~/falcon-monitor/<name>`
and `~/falcon-monitor/.previous/<name>` on the device — so a second rollback returns to the newer version —, loads the
restored monitor once with the device's Python, and runs `systemctl --user try-restart falcon-monitor-<name>.service`:
a user unit, no root, and `try-restart` restarts the unit only if it is running (it starts nothing that was stopped).
It refuses when the device has no previous version (a deploy before 4.2 kept none). `--dry-run` prints the commands
and sends nothing. Like deploy, SSH is used only when you run it.
## Measured

The per-step cost of the generated monitor, **on the development machine that ran `falcon monitor bench`, not on the
Orin** (x86-64, CPython 3.14, one core; 20 000 synthetic samples at 20 Hz, each signal a random walk with 2 in 100
samples unknown; `perf_counter_ns` around each step of the whole set):

| monitors | median | p95 | p99 | max |
|---|---|---|---|---|
| the 12 rules of `india` | 14 µs | 19–25 µs | 22–27 µs | 0.1–1.1 ms |
| one rule (`following_distance`) | 1.0 µs | 1.4 µs | 2.0 µs | up to 4.9 ms |
| one rule with a 10 s look-ahead (`keep_left`) | 2.9 µs | 4.3 µs | 5.1 µs | 38 µs |

Ranges are over three runs of the set; the single-rule rows are one run each. The maxima are single outliers of the
interpreter or the operating system (one step in 20 000), not the evaluator's steady cost — a pure-Python monitor
gives no bound on its worst case. At 20 Hz a step has 50 ms; the monitor's typical cost is small beside it on this
machine. The Orin's cores are slower and JetPack 6's Python is 3.10 (recalled, not checked on a device; the module
was only byte-compiled under 3.12 and run under 3.14 here): run `falcon monitor bench` there (copy the package, `python3 -c` with `falcon_monitor`'s
`MonitorSet`) before relying on a figure. The cost of the ROS 2 subscriptions and publications is not in these
numbers.

## Limits

- **Deployment is tested with a fake transport only.** No device was reached in the tests; the SSH path is
  `hil/deploy.py`'s, which the HIL deployment uses. The keep-previous step and the rollback were tested on a fake SSH
  layer that stands for the device's folders (tests/test_robustness42.py), not on a device: the shell commands they
  send (`cp -a`, `mv`, `systemctl --user try-restart`) were not run on JetPack.
- **The node was run against a fake rclpy** (plain classes standing in for the message types), not a ROS 2
  installation: message type names, QoS and the `byte` level of `DiagnosticStatus` follow ROS 2 Humble's interfaces
  as written from memory of them, not checked against an installation here.
- **The monitor is only as good as its signals.** The rule signals are Falcon's simulator-side definitions
  (`core.rules`: a straight corridor for `lead_gap_m`, the map's lane geometry for `lane_margin_m`, …). On the
  vehicle the stack must publish quantities that mean the same; Falcon does not generate that perception. An
  unmapped signal makes its rules report `unknown`, never a violation and never `ok`.
- A verdict that looks ahead is late by its horizon (10 s for the keep-side rule): it is a record, not a warning.
- Fixed-rate sampling of the latest values: a signal that changes and changes back between two samples is not seen.
- The monitor is evidence, not a safety mechanism: Falcon is not a qualified tool (ISO 26262-8 §11), and the
  generated code has not been developed to any safety standard.
