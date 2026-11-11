# The living safety case (GSN)

A safety argument in **Goal Structuring Notation** (GSN Community Standard v3) whose evidence is Falcon's own results,
**bound by hash and by claim** and re-verified whenever they change: a CI run that breaks a Clopper–Pearson bound turns
its Solution red, the red propagates to the top goal, and a CI gate can fail the build on it. Code:
`falcon_autonomy/core/gsn.py` (its docstring is the reference), `core/ci.py` (the re-verification and the gate),
`ui/gsn_tab.py` (Reports ▸ *Safety case (GSN)*).

> **Falcon Autonomy is not a qualified software tool in the sense of ISO 26262-8 §11.** No tool qualification (tool
> confidence level, validation, development to a safety standard) has been made. Its results are evidence for a safety
> team to assess, not a certification argument; their validity for the real vehicle is not shown by Falcon. Every
> template states this as a Justification on the top goal.

## Where it shows

| where | what |
|---|---|
| Reports ▸ *Safety case (GSN)* | the argument drawn in GSN shapes, each node coloured by its status as verified now; click a node for its text, status, why and evidence (result, claim, pinned SHA-256, when bound); *New from the template*, *Verify now*, *Re-bind* (undoable: *Undo* puts the previous evidence back), *Export HTML…* / *Export SVG…* |
| `falcon gsn new --project P [--template sotif\|iso26262\|blank] [--bind] [--force]` | start from a template; `--bind` pins every result it points at |
| `falcon gsn show --project P [--json]` | the tree with each node's status and why |
| `falcon gsn bind --project P NODE [--kind K --ref REF] [--measure M --bound B --conf C --limit L --data REF \| --claim JSON] [--unbind]` | bind or re-bind (keeping the claim when none is given) |
| `falcon gsn verify --project P [--json] [--gate] [--fail-on red\|amber]` | verify and log; `--gate` exits 1 when a top goal is red |
| `falcon gsn export --project P --out FILE.html\|FILE.svg` | the HTML report (also Reports ▸ *Safety case (GSN)* as a report kind, `falcon report gsn`) or an SVG |
| `falcon ci run` | re-verifies `ci:<name>@latest` evidence at the end of every run; `gates.gsn` fails the run on a red top goal ([CI](ci.md#the-living-safety-case)) |
| the assistant | `formal_gsn_tree` (read) and `formal_gsn_verify` (verify, optionally re-bind one Solution) |

## The document

`<workspace>/safety/gsn.json`, schema `falcon.gsn/1`:

```json
{"schema": "falcon.gsn/1", "title": "…", "template": "sotif",
 "nodes": [
   {"id": "G1", "type": "Goal", "text": "The residual risk … is acceptable"},
   {"id": "S1", "type": "Strategy", "text": "Argument over the scenario areas …"},
   {"id": "Sn1", "type": "Solution", "text": "CI aeb: its failure rate is bounded",
    "evidence": {"kind": "ci", "ref": "ci:aeb@latest",
                 "claim": {"measure": "failure_rate", "bound": "clopper_pearson_upper", "conf": 0.95, "limit": 0.05},
                 "sha256": "…", "path": "ci/aeb-1234/ci-results.json", "bound_at": "2026-11-12T10:00:00Z"}},
   {"id": "J1", "type": "Justification", "text": "Falcon Autonomy is not a qualified software tool …"}],
 "links": [{"from": "G1", "to": "S1", "kind": "SupportedBy"}, {"from": "G1", "to": "J1", "kind": "InContextOf"}]}
```

Node types: Goal, Strategy, Solution, Context, Assumption, Justification, AwayGoal (argued in another module:
`module`). `SupportedBy` goes from a Goal or Strategy to a Goal, Strategy, Solution or AwayGoal; `InContextOf` from a
Goal or Strategy to a Context, Assumption or Justification. The SupportedBy graph must be acyclic; the top goals are
the Goals nothing supports. A Goal or Strategy may be marked `"undeveloped": true`. Every load and save checks all of
this and names what is wrong.

### Evidence and claims

| kind | ref (relative to the workspace) | the file pinned |
|---|---|---|
| `failrate` | `studies/<estimate>` | `failure-rate.json` ([failure rate](failure-rate.md)) |
| `suite` | `suites/<suite>` | `results.jsonl` ([stack testing](stack-testing.md#suites)) |
| `ci` | `ci/<run>`, or the alias **`ci:<name>@latest`** — the newest CI run of that config name under `<workspace>/ci/` | `ci-results.json` |
| `falsify` | `studies/<falsification study>` | `falsify.json` ([falsification](falsification.md)) |
| `rules` | `rules/<check>` | `rules.json` ([traffic rules](traffic-rules.md)) |
| `lineage` | any file (a dataset's `data.json`, a model); a training folder `training/<id>` | the file itself; a training folder's `model.onnx` (its `data.json` when it has no model), a `model.json`'s registered model |

| claim `measure` | on | holds when |
|---|---|---|
| `failure_rate` | suite, ci, rules (violating runs), failrate | the `bound` of k failing in n runs is ≤ `limit`: `clopper_pearson_upper` (one-sided Clopper–Pearson at `conf`, default 0.95 — core.failrate's `binom_upper`), `clopper_pearson_lower`, or `estimate` (k/n); for an importance-sampling estimate `ci95_upper` (its 95 % upper bound; a refused estimate's zero-failure bound when it gives one) or `estimate` |
| `pass_rate` | suite, ci | `clopper_pearson_lower` (default) of the passing share ≥ `limit` |
| `min_robustness` | falsify | the best (lowest) ρ ≥ `limit` (0: never falsified within its budget) |
| `falsified_runs` | falsify | at most `limit` falsified runs |
| `violations_per_km` | rules, ci | violations per km ≤ `limit` |
| `verdict` | ci | the verdict equals `equals` (default `passed`) |
| `field` | any JSON | the number at the dotted `path` compared by `op` (`<=` `<` `>=` `>` `==`) with `limit`, or equal to `equals` (e.g. a lineage file's Merkle root) |
| `data_root` | lineage (a trained model) | the model names exactly the data bound in `data` (below) |
| (none) | any | the pinned hash alone |

**Binding** resolves the ref to its file, pins the file's SHA-256 and the time. An `@latest` alias pins its current
target too, but follows the newest result afterwards: that is what makes the case *living*.

#### Where it ran

A `failrate` or `falsify` result's evidence also records, when it is bound, **where its runs were made**
(`evidence.where`, from core.gsn's `where_ran`):

| `on` | from | recorded |
|---|---|---|
| `local` | runs without a `node` block | the number of runs |
| `cluster` | the `node` block of each run's run.json ([cluster](cluster.md#results)) | each node's name and its runs, the pools, the node kinds, the runs made here too, and every job's global id (`gids`, 20 hex digits each) |
| `cloud` | the study's `cloud.json`, written when a burst's summary comes back ([cloud](cloud.md#a-bursts-life)) | the provider, the burst's id, the GPU, the approved estimate's id and the instance |

Such a result is bound by hash and claim exactly as a local one. At every verification `where` is worked out again: a
result whose runs were since made elsewhere (a run driven again on another node) is **amber** until it is re-bound.
The Reports tab shows it as *Ran* (the jobs' ids in its tooltip), the HTML report under the result, and
`formal_gsn_tree` as `ran`.

#### A model's data: `data_root`

`falcon gsn bind NODE --kind lineage --ref training/<id> --measure data_root [--data REF]` — the claim that the
model's ONNX `metadata_props` `falcon.data_merkle_root` ([lineage](lineage.md#models)) is the root of the data in
`data`: by default the training's own `data.json`, or a **dataset version** (`datasets/<spec>/versions/vNNN.json`).
Binding pins the model (its `model.onnx`) and the data's files (`evidence.data.sha256`: the `data.json`, or the
version's JSON and its splits file). Verification:

- **amber** when the model changed since it was bound, the data changed or is missing (re-bind to accept it), or the
  check cannot be made (no `model.onnx`, the `onnx` package not installed, a version hashed with the other algorithm);
- **red** when the roots disagree — core.lineage's `verify_model` (without re-reading the images): the ONNX
  metadata against `data.json`'s roots, those roots against its per-sample addresses, the registration's SHA-256 —
  and, for another training's `data.json`, its training root against the model's; for a dataset version: the training
  names a different version (`falcon.dataset_version` / `data.json`'s `dataset_version`), a training frame is not a
  frame of the version's train split, or the version's splits file does not give the root it records;
- **green** otherwise.

A model's root and a version's root are never equal numbers: a version's leaves are every sensor's frames with
their **label files**, a model's the camera images it trained on with the labels **as trained** (after the class
selection and the ignore rule; [lineage](lineage.md#limits)). So against a version the claim checks what the roots
can show — the model's root is that of its samples, every sample's frame is in the version's train split, and the
training names the version — rather than comparing the two roots.

## Verification

| status | a Solution | a Goal or Strategy |
|---|---|---|
| **green** | its file's hash is the pinned one and the claim holds (or there is none) | every support green |
| **amber** | the file is missing; it changed since it was bound (*re-bind to accept it*); or the claim cannot be evaluated on it (a refused estimate's point, a field that is not there) | a support amber (and none red) |
| **red** | the claim is violated on the bound result — a bound broken | a support red |
| **grey** | no evidence bound (undeveloped) | marked undeveloped, supported by nothing, or a support grey (and none amber or red); an AwayGoal |

Propagation is AND, worst first: red > amber > grey > green. Context, Assumption and Justification nodes carry no
status. An alias is amber only when nothing resolves; a new CI result is not a "change" for it.

**The event log** `<workspace>/safety/gsn-events.jsonl` (`falcon.gsn_event/1`) is append-only — it is only ever
opened for appending: `new`, `bind` (with the previous evidence, so any re-bind can be undone by hand), `unbind`,
`verify` (each node's status, the top goals, who: cli, gui, ci or the assistant, and the aliases a CI run supplied),
`export`.

## Templates

- **ISO 21448 SOTIF** (default): G1 "the residual risk due to the intended functionality's insufficiencies is
  acceptable in the ODD", in the context of the ODD and the acceptance criteria, argued over the scenario areas:
  **known hazardous scenarios** (area 2) — a Solution per suite and CI config in the workspace (failure rate bounded
  by Clopper–Pearson at 0.95, limit 0.05 to edit) and the ODD coverage stated as Context (core.oddcoverage);
  **the unknown hazardous area** (area 3) — a Solution per falsification study (`min_robustness ≥ 0`) and per
  importance-sampling failure rate (`ci95_upper ≤ 1e-3` to edit), under the assumption that the parameters'
  declared distributions are how they occur; **triggering conditions** — a Context per study whose failures
  core.safetycase's `triggering_conditions` finds (the failing range of each parameter against the range tried),
  and the falsification study that found them. Where the workspace has no such result the goal is left undeveloped.
  A failure rate or falsification made on the cluster or a cloud burst says so in its Solution's text ("on the
  cluster (ws1 120, ws2 80; 200 jobs)", "on a cloud burst (runpod …)"). When the workspace has trained models that
  name their data (`training/<id>` with a `model.onnx` and a `data.json` with addresses), a goal **G7** "the
  machine-learned models are trained on the data stated for them, and neither changed since" holds a `data_root`
  Solution per model, bound to the dataset version its training names when that version is in the workspace, else to
  its own `data.json`.
- **ISO 26262**: G1 "the item is free of unreasonable risk due to hazards caused by malfunctioning behaviour of E/E
  systems", in the context of the item definition and the HARA; argued over the safety goals: the HARA and the safety
  concepts (undeveloped: outside Falcon), the verification evidence Falcon produces (suites, CI, rule checks) under the
  assumption that the simulation is representative, and hardware and software faults as an AwayGoal.
- **Blank**: one undeveloped goal and the Justification.

The templates are starting points; the argument, its wording and the limits are the safety team's to set.

## Measured

`tests/test_gsn.py`: a CI result with 0 failures in 60 runs bounds the failure rate at 1 − 0.05^(1/60) = 0.049 ≤ 0.1
(green, propagated to the top goal once the suite's Solution is bound too); a newer CI result with 4 failures in 60
lifts the bound to 0.146 — the `@latest` alias follows it and the Solution, its goal and the top goal turn red; a
re-run suite turns its Solution amber until it is re-bound; `falcon ci run` on the fake carla with the gate fails
(exit 1) once the claim's limit is one two runs cannot meet; the Reports tab is driven offscreen (verify, re-bind,
undo, export). `tests/test_gsn_evidence.py`: a failure rate estimated with `--on cluster` on two local worker daemons
(the fake carla) is bound with both nodes, their run counts and a global id per run, verifies green, turns amber when
one run's node block changes and green again once re-bound; a summary pulled back from a cloud burst records the
provider and the burst; three trained models (tiny ONNX files whose metadata core.training wrote) against a dataset
version: the one trained on it is green, one with a frame from the version's test split and one naming another
version are red; a changed ONNX root is amber, then red once re-bound; an edited version is amber.

## Limits

- A green node says only that the bound result shows what its claim states: not that the claim is the right one,
  that the evidence is sufficient, or that the simulation is valid for the vehicle.
- Clopper–Pearson treats the runs as independent draws of the same experiment; a suite's seeds and scenarios are a
  design, not a sample of the operating domain.
- Undo in the tab covers re-binds made in this session; the event log keeps every previous evidence for the rest.
- The drawing is a tidy tree from the top goals; a node with two parents is drawn under the first.
