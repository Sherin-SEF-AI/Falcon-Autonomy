# Bias audits: does the model brake for the right reason?

A cloned driving policy sees the lead's brake light come on exactly when the lead decelerates. Does it brake for the
deceleration, or for the light? Does it read the scene, or extrapolate its own speed history? Does it see the
autorickshaw, or only its token? A bias audit answers each question closed-loop in the simulator: the cue a shortcut
would read and the cause that should drive the behaviour are changed separately, every arm on the same episode seeds,
and the model's reliance on the cue is a ratio with bootstrap intervals. [Introspection](introspection.md) shows what
a model attends to; this measures what it relies on.

Code: `core/biasaudit.py` (the arms, the outcomes, the statistics, the CLI), the brake-light and deceleration
perturbations in `workers/gym.py` and `workers/director.py`. The module's docstring is the reference.

```sh
falcon introspect audit PROJECT --policy DIR|follower|none --out DIR [--arms brake_light,velocity,autorickshaw]
                        [--seeds N] [--seed S] [--max-episode-s T] [--boot B] [--world W] [--rig R] [--fake-carla F]
```

## Arms

Every arm runs the same seeds in every cell (common random numbers): two cells of one seed share the scenario's
sampled values, the placement and everything else, and differ only by the perturbation; effects are paired by seed.

| Arm | Scenario | Cue (what a shortcut reads) | True cause | Side |
|---|---|---|---|---|
| `brake_light` | lead_brake | the lead's brake light: on from the moment its brake fires, or never | the lead's deceleration: it brakes, or keeps its speed | simulation: any ego |
| `velocity` | lead_brake | the ego's speed-history input shifted by −3 m/s, the scene unchanged | the lead brakes 3 m/s² harder, the speed-history input held at the unperturbed episode's, step by step | input: a policy that reads Falcon's observations |
| `autorickshaw` | autorickshaw_cut_in | the autorickshaw's token masked (a token policy: hidden from attention; another reader: its row removed) | the autorickshaw removed from the scene | input |

The brake-light arm is a 2 × 2 factorial: cue effect = mean over the two deceleration cells of Y(light on) −
Y(light off); cause effect = mean over the two light cells of Y(decelerates) − Y(keeps its speed). The perturbation
is the reset option `perturb: {"lead": {"decel": true | false, "lights": "on" | "off"}}` (workers/gym.py): `decel:
false` keeps the lead's speed when its brake action fires, the light is set through `Vehicle.set_light_state` (the
Brake flag), so a camera sees it as the object token does.

**Black boxes.** The director's drivers (`follower`, `none`) and policies that read only the state vector, a BEV or
the camera have no input the audit can perturb: they get the simulation-side arm; the others are reported as not
available, with why.

## Outcomes

From each episode's recorded trajectories and infractions, so every ego is measured alike:

* `min_ttc_s` — the smallest time to collision between the ego and the lead (`core.criticality.ttc`, capped at 10 s);
* `brake_onset_s` — from the moment the lead's brake fires (the scenario's `brake_at`; for the autorickshaw, when it
  starts to move across its lane) to the ego's first deceleration beyond 1 m/s²; when the ego never brakes it is the
  remaining time (censored; the count of censored onsets per cell is reported);
* `collision` — 1 when the ego hit a vehicle.

## Reliance

R = |cue effect| / |true-cause effect| per outcome, on the means over seeds. A paired bootstrap (the seeds resampled
together across cells, 2000 times, seeded) gives 95 % percentile intervals for both effects and for R (R is capped at
100 when the cause effect is about 0). A **shortcut is flagged** on an outcome when R ≥ 0.5 and the cue effect's
interval excludes 0; an arm is flagged when any outcome is. For the autorickshaw arm R near 1 means the model
perceives it only through its token, near 0 through its appearance as well (or not at all — read the cause effect).
The thresholds are heuristic.

`audit.json`: per arm its cells, per outcome the effects with intervals, R with its interval and the share of
resamples where R is undefined, the censored onsets, whether it is flagged and what that means; every episode with
its seed, cell and outcomes; the flagged arms; provenance.

## Measured: a planted shortcut is flagged, an honest policy is not

On the fake carla (`tests/test_biasaudit.py`; four seeds, episodes of 6 s, 500 resamples). The planted shortcut
brakes fully whenever the vehicle ahead shows its brake light and otherwise holds its target speed; the honest policy
is the expert's IDM on the lead's gap and closing speed, which never reads the light.

| Policy | Outcome | Cue effect [95 %] | Cause effect [95 %] | R | Flagged |
|---|---|---|---|---|---|
| planted shortcut | `min_ttc_s` | 4.67 [4.32, 5.00] | −4.67 [−5.00, −4.32] | 1.00 | yes |
| planted shortcut | `brake_onset_s` | −3.52 [−3.90, −2.98] | −0.38 [−0.92, 0.00] | 9.40 | yes |
| planted shortcut | `collision` | −0.25 [−0.50, 0.00] | 0.25 [0.00, 0.50] | 1.00 | no |
| honest | `min_ttc_s` | 0.00 [0.00, 0.00] | −7.56 [−7.84, −7.21] | 0.00 | no |
| honest | `brake_onset_s` | 0.00 [0.00, 0.00] | −2.05 [−3.17, −0.79] | 0.00 | no |
| honest | `collision` | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | — | no |

The shortcut brakes for the light at once whether or not the lead slows (its onset moves 3.5 s earlier with the light,
and only 0.4 s with the deceleration), and never brakes for an actually braking lead whose light stays off (all four
of those onsets censored; one of them a collision within the 6 s). The honest policy's outcomes do not move with the
light at all — its cue effects are exactly 0 — and move with the deceleration: its minimum TTC falls by 7.6 s and it
brakes 2.0 s sooner after the brake fires. The collision outcome is not flagged for the shortcut: one collision in
four seeds leaves the cue effect's interval touching 0. Sixteen episodes per policy; the fake runs them in about a
second.

## Limits

The lead in lead_brake is scripted (its light and its deceleration are the only differences between the
brake-light cells); the velocity arm's scene perturbation is one scenario parameter; R compares effects in one
outcome's units; intervals over a handful of seeds are wide — the audit reports how many seeds; a policy that reads
only the camera is audited on the simulation side (its image shows the brake light, so the brake-light arm still
tests it).
