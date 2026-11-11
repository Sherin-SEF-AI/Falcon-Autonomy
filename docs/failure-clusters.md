# Failure clusters

A workspace collects failures in several places — the failing cases of rare-case searches and sweeps, the failed
runs of stack suites, the failed runs of CI. **Failure clustering** groups them into failure modes, names each mode
from what sets it apart, and after every CI run says which of its failures fall in **no known mode**: a new failure
mode. Code: `falcon_autonomy/core/failclusters.py` (numpy only; its docstring is the reference). Page: Triage ▸
Failure modes; commands: `falcon clusters`; the assistant's `research_clusters`.

```sh
falcon clusters build PROJECT [--sources search,suite,ci] [--cut 0.3] [--json]   # cluster, name, save
falcon clusters show PROJECT [CLUSTER] [--json]                                  # the saved clusters, or one
falcon clusters check PROJECT CI_DIR [--cut 0.3] [--json]                        # a CI run's new failure modes
```

## The failures

| source | a failure is | from |
|---|---|---|
| search | a study's run whose objective reached the study's failure threshold (its `search-*.json`, else 0.5) | `studies/<id>/results.jsonl` ([rare cases](rare-cases-mil.md)) |
| suite | a finished run that failed its pass criteria | `suites/<id>/results.jsonl` ([stack testing](stack-testing.md)) |
| ci | a finished run that failed | `ci/<name>/ci-runs.jsonl` ([CI mode](ci.md)) |

Each failure carries its scenario, parameter values, the **conditions** it ran in (from its run folder's `run.json`:
the world, fog density, sun altitude, precipitation, and the time of day named from the sun as the run index names it:
night below −6°, twilight below 6°, low sun below 20°), its **outcome** (collision, minimum TTC, criticality, the
names of the checks it failed, the RSS label when the study recorded one) and — when a "why did it fail?" answer
exists for its run ([root cause](root-cause.md), `rootcause/*/rootcause.json`) — the **factors whose removal fixed
it**. Runs that did not finish are not failures here (they are errors). At most the newest 2,000 are clustered.

## The distance

**Gower's general coefficient** (Gower 1971, *A general coefficient of similarity and some of its properties*,
Biometrics 27) over five groups of features; a group's distance is the mean of its features', the whole is the mean
of the groups both failures have (equal weights):

| group | features and their distance |
|---|---|
| scenario | the scenario's id: 0 the same, 1 not |
| parameters | each parameter value: a number \|a − b\| / its range over all failures; a choice 0 or 1 |
| conditions | world and time of day (0 or 1); fog, sun altitude, precipitation (range-normalised) |
| outcome | collision and RSS label (0 or 1); the failed checks (Jaccard distance of the two sets); minimum TTC and criticality (range-normalised) |
| cause | the root-cause factors that fixed it (Jaccard), only when both have an attribution |

A feature one of the two lacks is left out (Gower's rule for missing values), so a failure without a root-cause
answer is compared on the other four groups. Every distance is between 0 and 1.

## The clusters

**Agglomerative clustering with average linkage** (UPGMA) on that matrix, computed by the nearest-neighbour chain
(O(n²); exact for average linkage, which never inverts — checked against a brute-force agglomeration in the tests),
**cut at 0.3**: two groups merge while the average distance between their members is at most 0.3, so failures in one
mode differ, on average, in less than about a third of what they could. It is deterministic (ties to the lower index,
failures in time order). Each cluster has an id from its **medoid** (the member nearest on average to the others:
`C-` and 8 hex digits of its key, so a mode keeps its id while its medoid stays), its size, sources, scenarios,
**cohesion** (the mean distance within) and its distinguishing features.

## The names

From what sets a cluster apart from the other failures, computed, not written by a model:

- a **value** (a scenario, world, time of day, collision or not, an RSS label, a failed check, a cause) shared by at
  least 75 % of the cluster and at least 25 points more often than outside it;
- a **number** (a parameter, fog, sun, rain, TTC, criticality) whose median inside is at least a quarter of its range
  away from the median outside, shown as the cluster's 10th–90th percentile range, or as "≥" / "≤" when it sits at
  one end.

The name is three of them: the scenario; the best of where and how it ran (a parameter, a condition, a cause); the
best of how it failed (the outcome); on equal standing a value goes before a number — e.g. *lead_brake, night, fails
"no collision"* or *cut_in, in world-blr, fails "min TTC ≥ 1 s"* (the tests' planted modes). With none, *mixed
failures (n)*. **Name them with the assistant** (Triage, when the assistant is on) asks the fast model for a name an
engineer would say ("lead car braking at night in fog"); a name is kept only when every number in it is one of the
cluster's own (the [grounding check](assistant.md)), otherwise the computed name stands — and the computed name is
always in the tooltip and the saved file.

## New failure modes after a CI run

At the end of every `falcon ci run` whose project (or output folder, `WORKSPACE/ci/NAME`) names a workspace, each
failed run is compared with the clusters of **every other failure** of that workspace: when its average distance to
the nearest cluster's members is at most the cut, it is in that mode; otherwise it is **new**, and the new ones are
clustered among themselves — each such cluster is a new failure mode, named against everything known. The result is
`OUT/failure-modes.json` (`falcon.failure_modes/1`), a line in `summary.md` (**New failure modes**: how many, and
their names) and `failure_modes` in `ci-results.json`. The assistant's suggestions pick it up ("See the new failure
mode", on Stack testing, Triage and Home), and `research_clusters` with the CI run's id lists it.

## Saved clusters and the API

`falcon clusters build` (and Triage ▸ Cluster the failures) saves `WORKSPACE/clusters/clusters.json`
(`falcon.failure_clusters/1`: the method and its settings, every cluster, and an index of the failures by key),
written under the team workspace's [lock](team.md#the-lock). For other code — the research autopilot among it:

| function | returns |
|---|---|
| `collect(ws, sources, exclude, keep)` | the failures (dicts: `key`, `source`, `origin`, `index`, `run`, `scenario`, `seed`, `values`, `conditions`, `outcome`, `causes`, `present`, `time`, `commit`), oldest first |
| `distance_matrix(failures)` | the (n, n) Gower distances |
| `cluster(failures, cut, D)` | clusters, largest first: `id`, `name`, `size`, `members` (keys), `medoid`, `features`, `sources`, `scenarios`, `cohesion` |
| `describe(failures, inside, outside)` / `name_of(features, size)` | a group's distinguishing features / its name |
| `build(ws, sources, cut, keep, save)` / `load(ws)` | the saved document |
| `assign(new, reference, cut)` | per new failure its cluster (or None), the nearest and the distance |
| `check_ci(ws, ci_dir, records, cut, write, keep)` | a CI run's known and new failures and new modes |

The assistant clusters with `keep` leaving out failures whose run is private (from the dashcam footage), so they
shape no cluster and no name it sees.

## Tests

`tests/test_failclusters.py`: Gower's distance against a hand computation; the linkage against a brute-force
agglomeration on 12 random sets at three cuts; a workspace with two planted modes — a search's lead_brake failures at
night in fog with collisions plus a CI run's three like them, and a suite's jaywalker runs by day missing their TTC
check — found as two clusters of 13 and 6, named *lead_brake, night, fails "no collision"* and *jaywalker, day, fails
"min TTC ≥ 1 s"*, the same ids and names on a second build; a later CI run with two failures like the first mode
(known) and two cut-ins in the rain (one new mode named *cut_in, …*), also through CI mode's own hook and its summary
line; the commands; the assistant's tools (a private run left out), the suggestion, and a fake fast model's names —
one grounded and kept, one with an invented number rejected.

## Limits

- **Hand-chosen features, equal weights.** Two failures can be near for reasons that do not matter and far for ones
  that do; the groups are weighted equally because nothing measured says otherwise.
- **Parameters by name.** Two scenarios' parameters with the same name are compared as one (their ranges pooled).
- **One cut** for the whole workspace; a mode wider than 0.3 is split, two closer than it are merged.
- **Conditions from the config**, not from what the sensors saw (fog density as set, not visibility measured).
- **Names describe, they do not explain**: "night" in a name means the cluster's failures were at night far more than
  the others, not that night caused them; the cause group says what a root-cause answer found, when there is one.
- **A new failure mode is new to this workspace**, relative to the failures it holds; with few failures recorded,
  most look new.
