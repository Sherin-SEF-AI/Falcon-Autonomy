# Latency in the loop

A simulation step assumes the stack answers instantly: the control computed from the observation at *t* acts at *t*.
A real computer answers late, and sometimes much later. **Latency in the loop** feeds measured compute latency —
spikes included — back into closed-loop simulation, and finds each scenario's **latency budget**: the largest latency
at which it still passes. Code: `falcon_autonomy/core/latencyloop.py` (profiles, the delay line, the budget, the
command); the delay line is held by `workers/director.py`; the docstrings are the reference.

## Where it shows

| where | what |
|---|---|
| Hardware ▸ *Latency budget* | a latency profile's histogram (one axis in ms, the median and p99 marked, a tooltip per bar) with whether it was measured and on what; a budget's table — per scenario case the budget, where it first fails and its status (late only, fails anyway, holds); the cases that fail only when the computer is late; *Find the budget* queues the sweep |
| `falcon latency profile (--from-hil RUN [--measure M] \| --from-deploy DIR [--engine E] \| --from-file F \| --synthetic SPEC) (--out FILE \| --project P) [--name N] [--seed N]` | makes a profile |
| `falcon latency run PROJECT --scenario K (--profile FILE [--scale S] \| --constant MS) [--mode pipeline\|serial] [--fps F] [--seed N]` | one scenario run with the latency, beside the same run without |
| `falcon latency budget PROJECT --scenario K [--scenario K2 …] (--constant [--max-ms M] \| --profile FILE [--max-scale S --step D]) [--samples N] [--criterion auto\|passed\|criticality] [--threshold T] [--exhaustive] [--name N]` | the budget of each scenario case; writes `<workspace>/latency/<name>/latency-budget.json` and prints the table |
| `falcon latency show FILE` | a profile or a budget |
| the assistant | `analysis_latency_budget` (read the results), `analysis_latency_run` (queue the sweep) ([assistant tools](assistant-tools.md)) |

## Profiles

A profile (`falcon.latency_profile/1`) is an empirical distribution of latency in milliseconds: its sorted samples
(thinned to at most 1024 evenly spaced order statistics, minimum and maximum kept; the statistics are of every value
given) and **where it came from**:

| source | what is in it | measured? |
|---|---|---|
| `--from-hil RUN` | a [HIL session](hardware.md)'s `latency.jsonl`, one value per frame: by default `on_device` (camera received → control sent, on the device's own clock), or `e2e`, `processing`, `round_trip` | yes — on the session's device; a simulated device on this machine is said to be one |
| `--from-deploy DIR` | a [deployment](deploy.md)'s on-device profiling of one engine (`profile.json`: the timed runs after the warm-up) | yes — on the Orin, or on this machine's CPU when the device was simulated (said so). One model's inference only: pre- and post-processing are not in it |
| `--from-file F` | a JSON list, `{"samples_ms": […]}` or one number per line | as the file's author states; Falcon cannot know |
| `--synthetic constant:MS` · `lognormal:median=MS,sigma=S[,spike_p=P,spike_ms=MS][,n=N]` | a seeded synthetic distribution; a spike replaces a draw with probability `spike_p` | **no** |

Sampling is by inverse CDF with linear interpolation between order statistics, from a seeded generator: one uniform
number per simulation step. The same seed gives the same uniform at step *k* whatever the scale, so a run at scale 2
sees exactly twice the latency of the run at scale 1 at every step (common random numbers).

## The delay line

A run's config may carry a `latency` block (`{profile | constant_ms, scale, mode, seed}`); the profile's samples are
inside it, so the run is self-contained and its cache key follows the latency. **Without the block nothing changes**:
no delay line is created, the config and its cache key are what they were, and the run is the run it always was. The
tests drive the same scenario with no block and with a block of zero latency and compare the recorded frames byte for
byte.

At step *k* the stack or policy computes a control *u_k* from the observation at *t_k*. Its latency *L_k* = scale ×
(the constant, or a draw from the profile) is **quantised up to whole simulation steps**, *d_k* = ⌈*L_k* / dt⌉: a
result that arrives during a tick can only act from the next tick on. *L* = 0 gives *d* = 0 — the control acts in its
own step, exactly as without the delay line. The control applied at step *j* is the most recently *computed* one that
has arrived (*k* + *d_k* ≤ *j*); a result overtaken by a newer one is dropped. **Hold:** until a newer control
arrives the last applied one is held (zero-order hold); before the first arrives the vehicle coasts (throttle 0,
steer 0, brake 0).

| mode | meaning |
|---|---|
| `pipeline` (default) | every observation is processed; computations overlap, as in a pipelined stack. The staleness is the latency |
| `serial` | one computation at a time: the observations of the steps in between are skipped, so the staleness reaches up to twice the latency |

Where it acts — the narrowest place the closed loops share is the scenario director, which every scenario worker
builds:

| loop | wired |
|---|---|
| a scenario drive's ego (the director's lane follower; what sweeps, searches and the budget run) | yes |
| the learning environment's step (`workers/gym.py`: policies, RL, imitation rollouts) | yes |
| a stack run's follower baseline, and its `ros2` driver (on top of that driver's built-in one-step turnaround; scenario runs only) | yes |
| `workers/adversary.py`, the live session, a control replay, neural closed-loop replay (`recon/closedloop.py`) | **no** |
| HIL sessions | no — they have the real latency |

`run.json` → `scenario.latency` reports what was applied: the delay histogram in steps, the mean and maximum latency
drawn, the steps a control was held, the observations skipped (serial) and the results overtaken.

In `serial` mode the policy is still *called* at every step (its output is discarded while a computation is running),
so a policy with internal state advances it every step.

## The budget

For each scenario **case** (a scenario and one seeded parameter set; `--samples` cases per scenario) the latency *x*
is swept on a grid:

- `--constant`: a constant latency in whole steps, *x* = *d* · dt up to `--max-ms` — finer values are
  indistinguishable after quantisation;
- `--profile FILE`: the profile's **scale**, from 0 in steps of `--step` up to `--max-scale`; the budget is then a
  scale, reported with the profile's median, p99 and maximum at that scale in ms.

Every run of a case uses the same seed, parameters and uniform sequence (common random numbers). The case at *x* = 0
and at the maximum are run first; a case that passes at 0 and fails at the maximum is **bisected**
(`core.regbisect.Bisector`) to the largest *x* that still passes. The runs go through `sim.studio.Study` on
`sim.service.study_service`, cached by their config: a re-run resumes, and the run at *x* = 0 is the ordinary
scenario run.

| status | meaning |
|---|---|
| fails only when late | passes with no latency, fails from `first_failure` on: its budget is the last passing grid point |
| fails without latency | fails at *x* = 0: its failure is not the computer being late |
| no failure up to the maximum | the budget is *at least* the swept maximum |

The criterion (`--criterion`): `passed` — the scenario's own pass checks (`run.json` `scenario.passed`);
`criticality` — its [criticality](scenarios.md) below `--threshold` (1 is a collision); `auto` — the checks when the
scenario has them, else the criticality.

With `--constant --profile FILE` the constant-latency budget is read against the profile: per case, the share of the
profile's latencies that are over its budget.

**Bisection assumes failures are monotone in latency.** They need not be (a late brake can miss an obstacle that an
on-time brake would have stopped behind — and a later one still may pass it entirely). `--exhaustive` runs every grid
point and marks a case that passes above a failure as non-monotone.

## Measured (on the fake CARLA)

The tests run the budget on the fake CARLA server (kinematic motion, 10 Hz, so one step is 100 ms). For the built-in
`lead_brake` scenario with one seeded parameter set, a constant latency passes the scenario's own checks up to 800 ms
and fails from 900 ms; the bisection finds that in at most 7 runs instead of the 16 grid points and agrees with the exhaustive sweep,
and the run's criticality is higher at the larger latencies. These numbers describe the fake server's kinematic model and
Falcon's built-in follower, **not a real vehicle or stack**; they show the mechanism, nothing more.

## Limits

- **Step quantisation.** Every latency is rounded *up* to whole simulation steps: at 20 Hz, 1 ms and 50 ms are the
  same one-step delay, and a budget is only known to the step. Run at a higher `--fps` for a finer answer.
- **What a profile does not capture:** the correlation of consecutive latencies (samples are drawn independently;
  real spikes cluster, and a clustered spike is worse than scattered ones); thermal throttling and power-mode changes
  unless the profile was measured under them; the latency's dependence on the scene (more objects, slower); anything
  outside the measured interval (sensor exposure and readout, the actuator's own lag).
- A deployment profile is one model's inference; a HIL `on_device` profile is the device's whole turnaround in that
  session. Neither is the latency of a stack that was not the one measured.
- The delay is on the *control* path only: observations are not delayed separately, and the stack is assumed to know
  nothing of its own latency (no compensation).
- A real ROS 2 stack under `falcon stack` already takes real time, which the synchronous simulation hides by waiting
  for it; the delay line adds a simulated latency on top, it does not measure the stack's own.
- The tests use the fake CARLA; no real server was run for this feature.
