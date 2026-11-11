"""The living safety case: a Goal Structuring Notation (GSN) argument whose evidence is bound to Falcon's results by
hash and claim, re-verified whenever the evidence changes (standard library; docs/gsn.md is the manual).

The document (SCHEMA falcon.gsn/1, <workspace>/safety/gsn.json):

    {"schema": "falcon.gsn/1", "title", "template", "created",
     "nodes": [{"id", "type", "text", "undeveloped": bool, "module" (AwayGoal), "evidence" (Solution)}],
     "links": [{"from", "to", "kind": "SupportedBy" | "InContextOf"}]}

Node types (GSN Community Standard v3, 2021): Goal, Strategy, Solution, Context, Assumption, Justification and
AwayGoal (a goal argued in another module). SupportedBy links a Goal or Strategy to a Goal, Strategy, Solution or
AwayGoal; InContextOf links a Goal or Strategy to a Context, Assumption or Justification. The SupportedBy graph is
acyclic; a Solution supports nothing. The top goals are the Goals nothing supports.

Evidence. A Solution's evidence is {kind, ref, claim, sha256, path, bound_at}: kind (KINDS) and ref say which result
(a path relative to the workspace, or the alias "ci:<name>@latest": the newest CI run of that config name under
<workspace>/ci/); BINDING (bind()) resolves the ref to its file and pins the file's SHA-256; the claim is what the
result must show:

    failure_rate  k failing runs of n (suite, ci, rules: violating runs) bounded by clopper_pearson_upper (one-sided
                  Clopper–Pearson at conf, core.failrate.binom_upper) or clopper_pearson_lower or estimate (k/n); a
                  failrate result by ci95_upper (the importance-sampling estimate's 95 % upper bound; a refused
                  estimate's zero-failure bound when it gives one) or estimate — all ≤ limit
    pass_rate     passed of n bounded by clopper_pearson_lower (or _upper, estimate) ≥ limit
    min_robustness  a falsification study's best (lowest) ρ ≥ limit (core.stlobjective; e.g. 0: not falsified)
    falsified_runs  its falsified runs ≤ limit
    violations_per_km  a rule check's violations per km ≤ limit (core.rules)
    verdict       a CI run's verdict equals `equals` (default "passed")
    field         any number of the result's JSON at a dotted `path`, compared by `op` (<=, <, >=, >, ==) with limit
                  (or equal to `equals`) — e.g. a lineage file's Merkle root
    data_root     a trained model's data (kind lineage; ref the training folder or its model.onnx): the ONNX file's
                  metadata_props falcon.data_merkle_root is the root of the data the claim binds (`data`: the
                  training's own data.json, by default, or a dataset version's JSON) — core.lineage.verify_model's
                  checks (the ONNX metadata against data.json, data.json's roots against its per-sample addresses,
                  the registration's SHA-256), and for a version: the training names that version, every training
                  frame is in its train split and its splits file gives the root it records. Binding pins the model
                  and the data's files (data.sha256), so a changed model or changed data is amber; roots that
                  disagree are red
    (no claim)    the hash alone

Where it ran. Binding a failrate or falsify result also records where its runs were made (where_ran): on this machine,
on the cluster (each node's name and how many runs, the pools, the kinds and every job's global id, from the `node`
block of each run's run.json) or on a cloud burst (the provider, the burst's id, the GPU, the approved estimate's id
and the instance, from cloud.json in the study). It is checked again at every verification: a result whose runs were
since made elsewhere is amber until re-bound. A result from the cluster or a cloud burst is bound by hash and claim
exactly as a local one.

VERIFY (verify()) gives every node a status:
    green   a Solution whose file's hash is the pinned one and whose claim holds (or has no claim)
    amber   its file is missing, or changed since it was bound (re-bind to accept the new result), or its claim cannot
            be evaluated on it; an alias ("@latest") follows the newest result, so a new result is not a change —
            amber only when nothing resolves
    red     the claim is violated on the bound result (a bound broken: e.g. a CI run whose failures lift the
            Clopper–Pearson upper bound above the limit)
    grey    undeveloped: a Solution with no evidence bound, a Goal or Strategy marked undeveloped or supported by
            nothing, an AwayGoal (argued elsewhere)
and propagates it up the SupportedBy links as AND: a Goal or Strategy is red if any support is red, else amber if any
is amber, else grey if any is grey, else green. Context, Assumption and Justification nodes carry no status.

Templates: ISO 26262 (functional safety: hazards, safety goals, verification evidence) and ISO 21448 SOTIF (known
hazardous scenarios from the workspace's suites and ODD coverage; the unknown hazardous area from falsification
studies and importance-sampling failure rates; triggering conditions from core.safetycase.triggering_conditions).
Both state, as a Justification, that Falcon is NOT a qualified software tool in the sense of ISO 26262-8 §11 (no
tool qualification has been made): its results are evidence to be assessed, not a certification argument.

Every change and every verification is appended to <workspace>/safety/gsn-events.jsonl (EVENTS_SCHEMA; append-only:
never rewritten): new, bind, unbind, verify (with each node's status), export. core.ci re-verifies the CI aliases at
the end of every CI run and offers a gate (gates.gsn) that fails when a top goal is red.

    falcon gsn new --project P [--template iso26262|sotif|blank] [--bind] [--force]
    falcon gsn show --project P [--json]
    falcon gsn bind --project P NODE --kind K --ref REF [--measure M --bound B --conf C --limit L] | --unbind
    falcon gsn verify --project P [--json] [--gate]       --gate: exit 1 when a top goal is red
    falcon gsn export --project P --out FILE.html|FILE.svg

Limits: the argument's structure and wording are the user's (the templates are starting points, not a safety case);
a green node says only that the bound result shows what its claim states — not that the claim is the right one, the
evidence sufficient, or the simulation valid for the real vehicle.
"""
from __future__ import annotations

import copy
import json
import math
import os
import re
import sys

SCHEMA = 'falcon.gsn/1'
VERIFY_SCHEMA = 'falcon.gsn_verify/1'
EVENTS_SCHEMA = 'falcon.gsn_event/1'
TYPES = ('Goal', 'Strategy', 'Solution', 'Context', 'Assumption', 'Justification', 'AwayGoal')
LINKS = ('SupportedBy', 'InContextOf')
SUPPORTERS = ('Goal', 'Strategy')
SUPPORTED = ('Goal', 'Strategy', 'Solution', 'AwayGoal')
CONTEXTS = ('Context', 'Assumption', 'Justification')
KINDS = ('failrate', 'suite', 'ci', 'falsify', 'rules', 'lineage')
MEASURES = ('failure_rate', 'pass_rate', 'min_robustness', 'falsified_runs', 'violations_per_km', 'verdict', 'field',
            'data_root')
NO_LIMIT = ('verdict', 'data_root')            # measures that compare no number with a limit
WHERE_KINDS = ('failrate', 'falsify')          # results whose evidence records where their runs were made
BOUNDS = ('clopper_pearson_upper', 'clopper_pearson_lower', 'estimate', 'ci95_upper')
STATUSES = ('green', 'amber', 'red', 'grey')
RANK = {'green': 0, 'grey': 1, 'amber': 2, 'red': 3}
TEMPLATES = ('iso26262', 'sotif', 'blank')
NOT_QUALIFIED = ('Falcon Autonomy is not a qualified software tool in the sense of ISO 26262-8 §11 (no tool '
                 'qualification — tool confidence level, validation or development to a safety standard — has been '
                 'made): its simulation results are evidence to be assessed by the safety team, not a certification '
                 'argument, and their validity for the real vehicle is not shown by Falcon.')
_ALIAS = re.compile(r'^ci:(?P<name>[^@]+)@latest$')
_ID = re.compile(r'^[A-Za-z][\w.-]{0,31}$')


class GSNError(ValueError):
    pass


# ================================================================================================ files
def gsn_path(workspace: str) -> str:
    return os.path.join(workspace, 'safety', 'gsn.json')


def events_path(workspace: str) -> str:
    return os.path.join(workspace, 'safety', 'gsn-events.jsonl')


def load(workspace: str) -> dict:
    p = gsn_path(workspace)
    try:
        with open(p, encoding='utf-8') as fh:
            doc = json.load(fh)
    except FileNotFoundError:
        raise GSNError(f'no safety case at {p} (falcon gsn new)') from None
    except (OSError, ValueError) as e:
        raise GSNError(f'{p}: {e}') from None
    probs = validate(doc)
    if probs:
        raise GSNError(f'{p}: ' + '; '.join(probs[:5]))
    return doc


def save(workspace: str, doc: dict):
    from .provenance import write_json
    probs = validate(doc)
    if probs:
        raise GSNError('; '.join(probs[:5]))
    write_json(gsn_path(workspace), doc)


def doc_sha(doc: dict) -> str:
    from .provenance import sha256_json
    return sha256_json(doc)


def log_event(workspace: str, event: str, by: str = 'cli', **detail) -> dict:
    """Append one event to gsn-events.jsonl (append-only: the file is only ever opened for appending)."""
    from .provenance import utc_now
    rec = {'schema': EVENTS_SCHEMA, 'at': utc_now(), 'event': event, 'by': by, **detail}
    p = events_path(workspace)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'a', encoding='utf-8') as fh:
        fh.write(json.dumps(rec, ensure_ascii=False, default=str) + '\n')
    return rec


def events(workspace: str, limit: int = 0) -> list:
    p = events_path(workspace)
    if not os.path.isfile(p):
        return []
    with open(p, encoding='utf-8') as fh:
        out = [json.loads(x) for x in fh if x.strip()]
    return out[-limit:] if limit else out


# ================================================================================================ structure
def validate(doc: dict) -> list:
    """The document's problems (empty: valid): schema, ids, types, links (kinds, endpoints, what may link to what),
    a cycle of SupportedBy, evidence on a node that is not a Solution, a malformed evidence or claim."""
    out = []
    if not isinstance(doc, dict) or doc.get('schema') != SCHEMA:
        return [f'not a {SCHEMA} document']
    nodes = doc.get('nodes')
    links = doc.get('links')
    if not isinstance(nodes, list) or not isinstance(links, list):
        return ['nodes and links are lists']
    by = {}
    for n in nodes:
        nid = n.get('id') if isinstance(n, dict) else None
        if not isinstance(nid, str) or not _ID.match(nid):
            out.append(f'node id {nid!r}: a letter then letters, digits, _ . - (at most 32)')
            continue
        if nid in by:
            out.append(f'node {nid} appears twice')
        by[nid] = n
        if n.get('type') not in TYPES:
            out.append(f'node {nid}: type {n.get("type")!r} is not one of {", ".join(TYPES)}')
        if not isinstance(n.get('text', ''), str):
            out.append(f'node {nid}: text is a string')
        if n.get('evidence') is not None:
            if n.get('type') != 'Solution':
                out.append(f'node {nid}: only a Solution carries evidence')
            else:
                out += [f'node {nid}: {p}' for p in _evidence_problems(n['evidence'])]
    for k, ln in enumerate(links):
        a, b, kind = (ln or {}).get('from'), (ln or {}).get('to'), (ln or {}).get('kind')
        if kind not in LINKS:
            out.append(f'link {k}: kind {kind!r} is not one of {", ".join(LINKS)}')
            continue
        if a not in by or b not in by:
            out.append(f'link {k}: {a} → {b}: no such node')
            continue
        ta, tb = by[a]['type'], by[b]['type']
        if ta not in SUPPORTERS:
            out.append(f'link {a} → {b}: a {ta} supports nothing and has no context')
        elif kind == 'SupportedBy' and tb not in SUPPORTED:
            out.append(f'link {a} → {b}: a {tb} is not a support (use InContextOf)')
        elif kind == 'InContextOf' and tb not in CONTEXTS:
            out.append(f'link {a} → {b}: InContextOf leads to a Context, Assumption or Justification')
    if not out and _cycle(doc):
        out.append('the SupportedBy links form a cycle')
    return out


def _evidence_problems(ev) -> list:
    if not isinstance(ev, dict):
        return ['evidence is an object']
    out = []
    if ev.get('kind') not in KINDS:
        out.append(f'evidence kind {ev.get("kind")!r} is not one of {", ".join(KINDS)}')
    if not isinstance(ev.get('ref'), str) or not ev['ref']:
        out.append('evidence ref: the result (a path in the workspace, or ci:<name>@latest)')
    c = ev.get('claim')
    if c is not None:
        if not isinstance(c, dict) or c.get('measure') not in MEASURES:
            out.append(f'claim measure: one of {", ".join(MEASURES)}')
        else:
            if c.get('bound') is not None and c['bound'] not in BOUNDS:
                out.append(f'claim bound: one of {", ".join(BOUNDS)}')
            if c.get('conf') is not None and not 0 < float(c['conf']) < 1:
                out.append('claim conf lies in (0, 1)')
            if c['measure'] not in NO_LIMIT and c.get('limit') is None and c.get('equals') is None:
                out.append('claim: a limit (or equals)')
            if c['measure'] == 'data_root':
                if ev.get('kind') != 'lineage':
                    out.append('claim data_root: on lineage evidence (a trained model)')
                if c.get('data') is not None and (not isinstance(c['data'], str) or not c['data']):
                    out.append('claim data_root: data is the bound data\'s path (a data.json or a dataset version)')
            if c['measure'] == 'field' and not c.get('path'):
                out.append('claim field: its dotted path in the result')
            if c.get('op') is not None and c['op'] not in ('<=', '<', '>=', '>', '=='):
                out.append('claim op: <=, <, >=, > or ==')
    return out


def _cycle(doc) -> bool:
    kids = children(doc, 'SupportedBy')
    state = {}

    def visit(n):
        state[n] = 1
        for c in kids.get(n, ()):
            if state.get(c) == 1 or (state.get(c) is None and visit(c)):
                return True
        state[n] = 2
        return False
    return any(state.get(n['id']) is None and visit(n['id']) for n in doc['nodes'])


def children(doc: dict, kind: str | None = None) -> dict:
    out = {}
    for ln in doc['links']:
        if kind is None or ln['kind'] == kind:
            out.setdefault(ln['from'], []).append(ln['to'])
    return out


def node(doc: dict, nid: str) -> dict:
    for n in doc['nodes']:
        if n['id'] == nid:
            return n
    raise GSNError(f'no node {nid!r}')


def top_goals(doc: dict) -> list:
    supported = {ln['to'] for ln in doc['links'] if ln['kind'] == 'SupportedBy'}
    return [n['id'] for n in doc['nodes'] if n['type'] == 'Goal' and n['id'] not in supported]


def new_id(doc: dict, prefix: str) -> str:
    have = {n['id'] for n in doc['nodes']}
    k = 1
    while f'{prefix}{k}' in have:
        k += 1
    return f'{prefix}{k}'


# ================================================================================================ evidence
def latest_ci(workspace: str, name: str) -> str | None:
    """The newest CI output folder named `name` under <workspace>/ci (by its finished time, then its file's time)."""
    base = os.path.join(workspace, 'ci')
    best = None
    for d in sorted(os.listdir(base)) if os.path.isdir(base) else ():
        p = os.path.join(base, d, 'ci-results.json')
        try:
            with open(p, encoding='utf-8') as fh:
                r = json.load(fh)
        except (OSError, ValueError):
            continue
        if r.get('name') != name:
            continue
        k = (r.get('finished') or '', os.path.getmtime(p))
        if best is None or k > best[0]:
            best = (k, os.path.join(base, d))
    return best[1] if best else None


RESULT_FILE = {'failrate': 'failure-rate.json', 'suite': 'results.jsonl', 'ci': 'ci-results.json',
               'falsify': 'falsify.json', 'rules': 'rules.json'}


def resolve(workspace: str, ev: dict, overrides: dict | None = None) -> tuple:
    """(the evidence's file, its document or None, the alias it went through or None) — overrides: {ref: (path,
    document)} for a result not yet written (a CI run's own, at its end). (None, None, why) when nothing resolves."""
    ref = ev['ref']
    if overrides and ref in overrides:
        p, d = overrides[ref]
        return p, d, ref if _ALIAS.match(ref) else None
    alias = None
    m = _ALIAS.match(ref)
    if m:
        if ev['kind'] != 'ci':
            return None, None, f'{ref}: an @latest alias names a CI run (kind ci)'
        d = latest_ci(workspace, m.group('name'))
        if d is None:
            return None, None, f'no CI run named {m.group("name")!r} under {os.path.join(workspace, "ci")}'
        p, alias = os.path.join(d, 'ci-results.json'), ref
    else:
        p = ref if os.path.isabs(ref) else os.path.join(workspace, ref)
        if os.path.isdir(p) and ev['kind'] in RESULT_FILE:
            p = os.path.join(p, RESULT_FILE[ev['kind']])
        elif os.path.isdir(p) and ev['kind'] == 'lineage':        # a training folder: its model, else its data.json
            p = os.path.join(p, 'model.onnx') if os.path.isfile(os.path.join(p, 'model.onnx')) else \
                os.path.join(p, 'data.json')
        elif ev['kind'] == 'lineage' and os.path.basename(p) == 'model.json' and os.path.isfile(p):
            try:                                                   # a registration: the model it registers
                from .lineage import resolve_model
                p = resolve_model(p)['onnx'] or p
            except Exception:                                      # noqa: BLE001 (pinned as the file it is)
                pass
    if not os.path.isfile(p):
        return None, None, f'{p} is missing'
    return p, None, alias


def _read(p: str):
    if p.endswith('.jsonl'):
        with open(p, encoding='utf-8') as fh:
            return [json.loads(x) for x in fh if x.strip()]
    if p.endswith('.json'):
        with open(p, encoding='utf-8') as fh:
            return json.load(fh)
    return None


def _counts(kind: str, d) -> tuple:
    """(failing, passing-or-total…) as (k failures, n runs) of a result, or raise GSNError."""
    if kind == 'suite':
        rows = {}
        for r in d or ():
            if r.get('key'):
                rows[r['key']] = r
        ok = [r for r in rows.values() if r.get('status') == 'ok' and r.get('passed') is not None]
        return sum(1 for r in ok if r['passed'] is False), len(ok)
    if kind == 'ci':
        o = ((d.get('summary') or {}).get('overall') or {})
        if 'runs' not in o:
            raise GSNError('the CI result has no run summary')
        return int(o.get('failed') or 0), int(o['runs'])
    if kind == 'rules':
        res = d.get('results') or []
        return sum(1 for r in res if (r.get('totals') or {}).get('violated_rules')), len(res)
    raise GSNError(f'a {kind} result has no run counts')


def _bounded(k: int, n: int, bound: str, conf: float) -> float | None:
    from .failrate import binom_lower, binom_upper
    if n <= 0:
        return None
    if bound == 'clopper_pearson_upper':
        return binom_upper(k, n, conf)
    if bound == 'clopper_pearson_lower':
        return binom_lower(k, n, conf)
    if bound == 'estimate':
        return k / n
    raise GSNError(f'bound {bound} does not apply to run counts')


def _cmp(v, op: str, limit) -> bool:
    return {'<=': v <= limit, '<': v < limit, '>=': v >= limit, '>': v > limit, '==': v == limit}[op]


def _field(d, path: str):
    cur = d
    for part in path.split('.'):
        if isinstance(cur, list) and part.lstrip('-').isdigit():
            cur = cur[int(part)]
        elif isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            raise GSNError(f'the result has no {path}')
    return cur


def evaluate_claim(kind: str, claim: dict | None, d) -> dict:
    """{'ok': True | False | None (cannot be evaluated), 'measured', 'detail'} of a claim on a result document."""
    if not claim:
        return {'ok': True, 'measured': None, 'detail': 'no claim: the pinned hash alone'}
    m, conf = claim['measure'], float(claim.get('conf', 0.95))
    lim = claim.get('limit')
    try:
        if m in ('failure_rate', 'pass_rate'):
            bound = claim.get('bound') or ('ci95_upper' if kind == 'failrate' else 'clopper_pearson_upper'
                                           if m == 'failure_rate' else 'clopper_pearson_lower')
            if kind == 'failrate':
                if m != 'failure_rate':
                    raise GSNError('a failure-rate estimate has no pass rate')
                e = d.get('estimate') or {}
                if bound == 'estimate':
                    v = e.get('p') if e.get('status') in ('ok', 'warning') else None
                elif bound == 'ci95_upper':
                    v = (e.get('ci95') or [None, None])[1] if e.get('status') in ('ok', 'warning') else e.get('upper95')
                else:
                    raise GSNError(f'an importance-sampling estimate is bounded by ci95_upper or estimate, not {bound}')
                if v is None:
                    return {'ok': None, 'measured': None, 'detail': f'the estimate was {e.get("status")}: '
                            + '; '.join(e.get('reasons') or ()) + ' (no bound to compare)'}
                ok = float(v) <= float(lim)
                return {'ok': ok, 'measured': float(v), 'detail': f'{bound} {float(v):.3g} '
                        f'{"≤" if ok else ">"} {float(lim):g} ({e.get("failures")} failing of {e.get("n")} final runs, '
                        f'status {e.get("status")})'}
            k, n = _counts(kind, d)
            if m == 'pass_rate':
                k = n - k
            v = _bounded(k, n, bound, conf)
            if v is None:
                return {'ok': None, 'measured': None, 'detail': 'no finished runs'}
            ok = v <= float(lim) if m == 'failure_rate' else v >= float(lim)
            what = 'failing' if m == 'failure_rate' else 'passing'
            return {'ok': ok, 'measured': v, 'k': k, 'n': n,
                    'detail': f'{k} {what} of {n} runs: {bound.replace("_", " ")} at {conf:g} = {v:.4g} '
                              f'{("≤" if ok else ">") if m == "failure_rate" else ("≥" if ok else "<")} '
                              f'{float(lim):g}'}
        if m in ('min_robustness', 'falsified_runs'):
            if kind != 'falsify':
                raise GSNError(f'{m} is a falsification study\'s measure')
            s = d.get('summary') or {}
            if m == 'min_robustness':
                from .stlobjective import fnum
                v = fnum(s.get('best_rho'))
                if math.isnan(v):
                    return {'ok': None, 'measured': None, 'detail': 'no run\'s robustness was known'}
                ok = v >= float(lim)
                return {'ok': ok, 'measured': s.get('best_rho'), 'detail': f'best ρ {s.get("best_rho")} '
                        f'{"≥" if ok else "<"} {float(lim):g} over {s.get("known")} runs'}
            v = len(s.get('falsified') or ())
            ok = v <= float(lim)
            return {'ok': ok, 'measured': v, 'detail': f'{v} falsified runs of {s.get("evaluated")} '
                    f'{"≤" if ok else ">"} {float(lim):g}'}
        if m == 'violations_per_km':
            if kind == 'rules':
                res = d.get('results') or []
                km = sum(float(r.get('distance_km') or 0) for r in res)
                vio = sum(int((r.get('totals') or {}).get('violations') or 0) for r in res)
            elif kind == 'ci':
                row = (((d.get('summary') or {}).get('overall') or {}).get('metrics') or {}).get(
                    'rule_violations_per_km') or {}
                km, vio = float(row.get('distance_km') or 0), float(row.get('violations') or 0)
            else:
                raise GSNError(f'a {kind} result has no violations per km')
            if km <= 0:
                return {'ok': None, 'measured': None, 'detail': 'no distance driven'}
            v = vio / km
            ok = v <= float(lim)
            return {'ok': ok, 'measured': v, 'detail': f'{vio:g} violations over {km:.3f} km = {v:.3g} per km '
                    f'{"≤" if ok else ">"} {float(lim):g}'}
        if m == 'verdict':
            want = claim.get('equals', 'passed')
            v = d.get('verdict')
            return {'ok': v == want, 'measured': v, 'detail': f'verdict {v} ({"is" if v == want else "is not"} '
                                                              f'{want})'}
        if m == 'field':
            v = _field(d, claim['path'])
            if claim.get('equals') is not None:
                ok = v == claim['equals']
                return {'ok': ok, 'measured': v, 'detail': f'{claim["path"]} = {v!r} ({"=" if ok else "≠"} '
                        f'{claim["equals"]!r})'}
            op = claim.get('op', '<=')
            ok = _cmp(float(v), op, float(lim))
            return {'ok': ok, 'measured': v, 'detail': f'{claim["path"]} = {v} {op if ok else "not " + op} '
                                                       f'{float(lim):g}'}
    except GSNError as e:
        return {'ok': None, 'measured': None, 'detail': str(e)}
    except (KeyError, TypeError, ValueError, IndexError) as e:
        return {'ok': None, 'measured': None, 'detail': f'the result cannot be read for this claim ({e})'}
    return {'ok': None, 'measured': None, 'detail': f'unknown measure {m}'}


def _sha(p: str) -> str:
    from .provenance import sha256_file
    return sha256_file(p)


# ------------------------------------------------------------------------------------------------ where it ran
def where_ran(study_dir: str) -> dict:
    """Where a study's runs were made (module doc, "Where it ran"): {'on': 'cloud', 'provider', 'burst', 'gpu',
    'estimate_id', 'instance_id'} from the study's cloud.json; {'on': 'cluster', 'nodes': {name: runs}, 'pools',
    'kinds', 'runs', 'local_runs', 'gids'} when any of its runs' run.json has a `node` block; else {'on': 'local',
    'runs': n}. Its runs: results.jsonl's and failure-rate.json's final runs."""
    cj = os.path.join(study_dir, 'cloud.json')
    if os.path.isfile(cj):
        try:
            d = _read(cj) or {}
        except (OSError, ValueError):
            d = {}
        return dict({'on': 'cloud'}, **{k: d[k] for k in ('provider', 'burst', 'gpu', 'estimate_id', 'instance_id')
                                         if d.get(k)})
    runs = set()
    ws = os.path.dirname(os.path.dirname(os.path.abspath(study_dir)))
    try:
        for r in _read(os.path.join(study_dir, 'results.jsonl')) or ():
            if isinstance(r, dict) and r.get('run'):
                runs.add(r['run'])
    except (OSError, ValueError):
        pass
    try:
        for r in (_read(os.path.join(study_dir, 'failure-rate.json')) or {}).get('final_runs') or ():
            if isinstance(r, dict) and r.get('run'):
                runs.add(r['run'])
    except (OSError, ValueError, AttributeError):
        pass
    nodes, pools, kinds, gids, local = {}, set(), set(), set(), 0
    for run in sorted(runs):
        run = run if os.path.isabs(run) else os.path.join(ws, run)
        try:
            with open(os.path.join(run, 'run.json'), encoding='utf-8') as fh:
                n = json.load(fh).get('node')
        except (OSError, ValueError, AttributeError):
            continue
        if isinstance(n, dict) and n.get('name'):
            nodes[n['name']] = nodes.get(n['name'], 0) + 1
            pools.add(str(n.get('pool') or 'default'))
            kinds.add(str(n.get('kind') or ''))
            gid = str(n.get('dispatch') or '').split('.', 1)[0]
            if gid:
                gids.add(gid)
        else:
            local += 1
    if not nodes:
        return {'on': 'local', 'runs': local}
    return {'on': 'cluster', 'nodes': dict(sorted(nodes.items())), 'pools': sorted(pools), 'kinds': sorted(kinds),
            'runs': sum(nodes.values()), 'local_runs': local, 'gids': sorted(gids)}


def where_text(w: dict | None) -> str:
    """Where it ran, in a few words (the tab, the report, the template's texts)."""
    if not w:
        return ''
    if w.get('on') == 'cloud':
        return 'on a cloud burst' + (f' ({w["provider"]}' + (f' {w["gpu"]}' if w.get('gpu') else '') +
                                     (f', burst {w["burst"]}' if w.get('burst') else '') + ')'
                                     if w.get('provider') else '')
    if w.get('on') == 'cluster':
        names = ', '.join(f'{k} {v}' for k, v in (w.get('nodes') or {}).items())
        return (f'on the cluster ({names}' + (f'; {w["local_runs"]} here' if w.get('local_runs') else '') +
                f'; {len(w.get("gids") or ())} jobs)')
    return 'on this machine'


# ------------------------------------------------------------------------------------------------ a model's data
def _rel(workspace: str, p: str) -> str:
    """A path as evidence keeps it: relative to the workspace when it is inside it."""
    ap = os.path.abspath(p)
    return os.path.relpath(ap, workspace) if ap.startswith(os.path.abspath(workspace) + os.sep) else ap


def _data_files(workspace: str, model_path: str, claim: dict) -> dict:
    """{'ref', 'kind': 'training' | 'version', 'files': [absolute paths whose bytes the data is]} of a data_root
    claim's data: its `data` (a dataset version's JSON: the JSON and its splits file; a data.json or its training
    folder), or the model's own data.json."""
    ref = claim.get('data') or ''
    if ref:
        p = ref if os.path.isabs(ref) else os.path.join(workspace, ref)
    else:
        p = os.path.join(os.path.dirname(model_path), 'data.json')
        ref = _rel(workspace, p)
    if os.path.isdir(p):
        p = os.path.join(p, 'data.json')
    if not os.path.isfile(p):
        raise GSNError(f'the data {ref} is missing')
    try:
        doc = _read(p) if p.endswith('.json') else None
    except (OSError, ValueError) as e:
        raise GSNError(f'the data {ref} cannot be read: {e}') from None
    if isinstance(doc, dict) and doc.get('schema') == 'falcon.dataset_version/1':
        lin = doc.get('lineage') or {}
        if not lin.get('splits'):
            raise GSNError(f'{ref}: the dataset version has no lineage (made before lineage, or without pyarrow)')
        vws = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(p)))))
        return {'ref': ref, 'kind': 'version', 'files': [p, os.path.join(vws, lin['splits'])]}
    if isinstance(doc, dict) and (doc.get('train') is not None or doc.get('hash_alg')):
        return {'ref': ref, 'kind': 'training', 'files': [p]}
    raise GSNError(f'{ref}: neither a training\'s data.json nor a dataset version')


def _pin_data(workspace: str, model_path: str, claim: dict) -> dict:
    d = _data_files(workspace, model_path, claim)
    return {'ref': d['ref'], 'kind': d['kind'], 'sha256': {_rel(workspace, f): _sha(f) for f in d['files']}}


def _data_changed(workspace: str, pinned: dict) -> str:
    """Why the bound data is no longer what was pinned ('' when it is)."""
    for rel, sha in (pinned.get('sha256') or {}).items():
        f = rel if os.path.isabs(rel) else os.path.join(workspace, rel)
        if not os.path.isfile(f):
            return f'the data\'s {rel} is missing'
        now = _sha(f)
        if now != sha:
            return f'the data changed since it was bound ({rel}: {sha[:12]} → {now[:12]}): re-bind to accept it'
    return ''


def _version_label(doc: dict) -> str:
    """How a training names a dataset version (core.training.dataset_version_of)."""
    return f'{doc.get("name")} v{doc.get("version")} ({doc.get("spec")} sha256 {str(doc.get("sha256"))[:12]})'


def data_root_claim(workspace: str, model_path: str, claim: dict) -> dict:
    """evaluate_claim's answer for a data_root claim (module doc): ok True when the model names exactly the bound
    data, False when the roots (or the version's name, or its frames) disagree, None when it cannot be told."""
    from . import lineage as L
    if not model_path.endswith('.onnx'):
        return {'ok': None, 'measured': None, 'detail': 'no model.onnx: there is no root in a model to compare'}
    try:
        d = _data_files(workspace, model_path, claim)
        chk = L.verify_model(model_path, reread=False)
    except (GSNError, L.LineageError) as e:
        return {'ok': None, 'measured': None, 'detail': str(e)}
    except ImportError as e:
        return {'ok': None, 'measured': None, 'detail': f'the ONNX metadata cannot be read here ({e.name} is not '
                                                        'installed)'}
    except (OSError, ValueError) as e:
        return {'ok': None, 'measured': None, 'detail': f'the model cannot be read: {e}'}
    onnx_root = (chk['checked'].get('onnx_metadata') or {}).get('falcon.data_merkle_root')
    problems = list(chk['problems'])
    if d['kind'] == 'training':
        mine = os.path.realpath(os.path.join(chk['model']['dir'], 'data.json'))
        if os.path.realpath(d['files'][0]) != mine:          # another training's data: compare with its root
            other = _read(d['files'][0])
            alg = other.get('hash_alg')
            rows = (other.get('train') or {}).get('images') or []
            want = L.pairs_root(((r.get('frame'), r.get('label')) for r in rows), alg) if alg else None
            if want != onnx_root:
                problems.append(f'the model\'s falcon.data_merkle_root {onnx_root} is not {d["ref"]}\'s training '
                                f'root {want}')
        if problems:
            return {'ok': False, 'measured': onnx_root, 'detail': '; '.join(problems)}
        return {'ok': True, 'measured': onnx_root, 'detail': f'the model\'s falcon.data_merkle_root {onnx_root} is '
                f'the root of {d["ref"]}\'s training samples'}
    vdoc = _read(d['files'][0])
    lin = vdoc.get('lineage') or {}
    try:
        rows, meta = L.read_table(d['files'][1])
    except (L.LineageError, OSError, ValueError) as e:
        return {'ok': None, 'measured': onnx_root, 'detail': f'the version\'s splits cannot be read: {e}'}
    alg = meta.get('hash_alg') or lin.get('hash_alg')
    vroot = L.pairs_root(((r['frame'], r['label']) for r in rows), alg)
    if vroot != lin.get('merkle_root'):
        problems.append(f'the version\'s splits give root {vroot}, it records {lin.get("merkle_root")}')
    tdoc = _read(os.path.join(chk['model']['dir'], 'data.json'))
    label = _version_label(vdoc)
    if label not in str(tdoc.get('dataset_version') or ''):
        problems.append(f'the model names its data as {tdoc.get("dataset_version") or "none"!r}, not {label!r}')
    if tdoc.get('hash_alg') != alg:
        return {'ok': None, 'measured': onnx_root, 'detail': f'the model is hashed with {tdoc.get("hash_alg")} and '
                f'the version with {alg}: they cannot be compared'}
    split = 'train' if any(r['split'] == 'train' for r in rows) else None
    have = {r['frame'] for r in rows if split is None or r['split'] == split}
    frames = [r.get('frame') for r in (tdoc.get('train') or {}).get('images') or ()]
    outside = sum(1 for f in frames if f not in have)
    if outside:
        problems.append(f'{outside} of the model\'s {len(frames)} training frames are not in the version\'s '
                        f'{split or "samples"}')
    if problems:
        return {'ok': False, 'measured': onnx_root, 'detail': '; '.join(problems)}
    return {'ok': True, 'measured': onnx_root, 'detail': f'the model\'s falcon.data_merkle_root {onnx_root} is the '
            f'root of its {len(frames)} training samples, every one a frame of {label}\'s {split or "samples"} '
            f'(version root {vroot})'}


def check_evidence(workspace: str, ev: dict, overrides: dict | None = None) -> dict:
    """A Solution's evidence → {'status', 'why', 'path', 'sha256', 'alias', 'claim': evaluate_claim(...)}."""
    p, d, alias = resolve(workspace, ev, overrides)
    if p is None:
        return {'status': 'amber', 'why': f'evidence missing: {alias}', 'path': None, 'sha256': None}
    out = {'path': os.path.relpath(p, workspace) if not os.path.isabs(ev['ref']) else p, 'alias': alias}
    if d is None:
        sha = _sha(p)
        out['sha256'] = sha
        if not alias and ev.get('sha256') and sha != ev['sha256']:
            return dict(out, status='amber', why=f'the result changed since it was bound ({ev["sha256"][:12]} → '
                                                 f'{sha[:12]}): re-bind to accept it')
        if not alias and not ev.get('sha256'):
            return dict(out, status='grey', why='not bound yet (no hash pinned)')
        try:
            d = _read(p)
        except (OSError, ValueError) as e:
            return dict(out, status='amber', why=f'the result cannot be read: {e}')
    else:
        out['sha256'] = None                       # an in-memory result (a CI run at its end)
    if ev.get('where') is not None:
        out['where'] = ev['where']
        now = where_ran(os.path.dirname(p))
        if now != ev['where'] and not alias:
            return dict(out, status='amber', where_now=now,
                        why=f'its runs were made {where_text(now)} now, {where_text(ev["where"])} when bound: re-bind '
                            'to accept it')
    claim = ev.get('claim') or {}
    if ev.get('data'):
        why = _data_changed(workspace, ev['data'])
        if why:
            return dict(out, status='amber', why=why)
    if ev['kind'] == 'lineage' and not claim:
        return dict(out, status='green', why='the pinned hash matches', claim=None)
    if claim.get('measure') == 'data_root':
        c = data_root_claim(workspace, p, claim)
    else:
        c = evaluate_claim(ev['kind'], ev.get('claim'), d)
    out['claim'] = c
    if c['ok'] is None:
        return dict(out, status='amber', why=f'the claim cannot be evaluated: {c["detail"]}')
    return dict(out, status='green' if c['ok'] else 'red', why=c['detail'])


def bind(workspace: str, doc: dict, nid: str, kind: str, ref: str, claim: dict | None = None,
         keep_claim: bool = False) -> dict:
    """Bind evidence to a Solution: resolve the ref, pin its file's SHA-256 (the alias's current target for an
    @latest ref); returns the new evidence (doc changed in place; the caller saves and logs). keep_claim: re-bind
    with the node's current claim."""
    from .provenance import utc_now
    n = node(doc, nid)
    if n['type'] != 'Solution':
        raise GSNError(f'{nid} is a {n["type"]}: evidence binds to a Solution')
    if keep_claim and claim is None:
        claim = (n.get('evidence') or {}).get('claim')
    ev = {'kind': kind, 'ref': ref, 'claim': claim}
    probs = _evidence_problems(ev)
    if probs:
        raise GSNError('; '.join(probs))
    p, _d, why = resolve(workspace, ev)
    if p is None:
        raise GSNError(f'cannot bind {nid}: {why}')
    ev.update(sha256=_sha(p), path=os.path.relpath(p, workspace), bound_at=utc_now())
    if kind in WHERE_KINDS:
        ev['where'] = where_ran(os.path.dirname(p))
    if (claim or {}).get('measure') == 'data_root':
        ev['data'] = _pin_data(workspace, p, claim)
    n['evidence'] = ev
    n.pop('undeveloped', None)
    return ev


def verify(workspace: str, doc: dict, overrides: dict | None = None) -> dict:
    """Every node's status (module doc) with why, the top goals' and the counts."""
    from .provenance import utc_now
    by = {n['id']: n for n in doc['nodes']}
    kids = children(doc, 'SupportedBy')
    res = {}

    def status(nid):
        if nid in res:
            return res[nid]['status']
        n = by[nid]
        t = n['type']
        if t in CONTEXTS:
            res[nid] = {'status': None, 'why': t.lower()}
        elif t == 'AwayGoal':
            res[nid] = {'status': 'grey', 'why': f'argued in another module ({n.get("module") or "not named"})'}
        elif t == 'Solution':
            ev = n.get('evidence')
            res[nid] = ({'status': 'grey', 'why': 'no evidence bound'} if not ev
                        else check_evidence(workspace, ev, overrides))
        else:
            sup = kids.get(nid, [])
            if n.get('undeveloped') or not sup:
                res[nid] = {'status': 'grey', 'why': 'undeveloped' if n.get('undeveloped') else 'supported by nothing'}
                for c in sup:                         # still give its supports a status
                    status(c)
            else:
                st = [(status(c), c) for c in sup]
                worst = max(st, key=lambda x: RANK[x[0]])
                which = [c for s_, c in st if s_ == worst[0]]
                why = {'green': f'all {len(st)} supports hold', 'red': f'violated below: {", ".join(which)}',
                       'amber': f'evidence to re-check below: {", ".join(which)}',
                       'grey': f'undeveloped below: {", ".join(which)}'}[worst[0]]
                res[nid] = {'status': worst[0], 'why': why}
        return res[nid]['status']
    for n in doc['nodes']:
        status(n['id'])
    tops = {g: res[g]['status'] for g in top_goals(doc)}
    counts = {s: sum(1 for r in res.values() if r['status'] == s) for s in STATUSES}
    return {'schema': VERIFY_SCHEMA, 'at': utc_now(), 'nodes': res, 'top': tops, 'counts': counts,
            'gsn_sha256': doc_sha(doc)}


def verify_and_log(workspace: str, by: str = 'cli', overrides: dict | None = None, doc: dict | None = None) -> dict:
    doc = load(workspace) if doc is None else doc
    v = verify(workspace, doc, overrides)
    log_event(workspace, 'verify', by, top=v['top'], counts=v['counts'], gsn_sha256=v['gsn_sha256'],
              statuses={k: x['status'] for k, x in v['nodes'].items() if x['status']},
              aliases=sorted(overrides or ()))
    return v


def gate(v: dict, goals=None, fail_on: str = 'red') -> tuple:
    """(passed, the top goals judged with their status) — failed when one is red (or amber too: fail_on amber)."""
    bad = ('red',) if fail_on == 'red' else ('red', 'amber')
    tops = {g: s for g, s in v['top'].items() if not goals or g in goals}
    return not any(s in bad for s in tops.values()), tops


# ================================================================================================ templates
def _n(nid, typ, text, **kw):
    return dict({'id': nid, 'type': typ, 'text': text}, **kw)


def _l(a, b, kind='SupportedBy'):
    return {'from': a, 'to': b, 'kind': kind}


def evidence_candidates(workspace: str) -> dict:
    """What a template can point Solutions at: {kind: [ref]} — suites, CI configs (as @latest aliases), failure-rate
    estimates, falsification studies, rule checks."""
    out = {k: [] for k in KINDS}

    def ls(*p):
        d = os.path.join(workspace, *p)
        return sorted(os.listdir(d)) if os.path.isdir(d) else []
    for s in ls('suites'):
        if os.path.isfile(os.path.join(workspace, 'suites', s, 'results.jsonl')):
            out['suite'].append(f'suites/{s}')
    names = []
    for c in ls('ci'):
        try:
            with open(os.path.join(workspace, 'ci', c, 'ci-results.json'), encoding='utf-8') as fh:
                nm = json.load(fh).get('name')
        except (OSError, ValueError):
            continue
        if nm and nm not in names:
            names.append(nm)
    out['ci'] = [f'ci:{nm}@latest' for nm in names]
    for s in ls('studies'):
        if os.path.isfile(os.path.join(workspace, 'studies', s, 'failure-rate.json')):
            out['failrate'].append(f'studies/{s}')
        if os.path.isfile(os.path.join(workspace, 'studies', s, 'falsify.json')):
            out['falsify'].append(f'studies/{s}')
    for s in ls('rules'):
        if os.path.isfile(os.path.join(workspace, 'rules', s, 'rules.json')):
            out['rules'].append(f'rules/{s}')
    for t in ls('training'):                           # trained models that name their data (core.lineage)
        d = os.path.join(workspace, 'training', t)
        if os.path.isfile(os.path.join(d, 'model.onnx')):
            try:
                with open(os.path.join(d, 'data.json'), encoding='utf-8') as fh:
                    if json.load(fh).get('hash_alg'):
                        out['lineage'].append(f'training/{t}')
            except (OSError, ValueError, AttributeError):
                continue
    return out


def model_data(workspace: str, ref: str) -> str:
    """What a template binds a model's data_root claim to: the dataset version its training names (found in the
    workspace's datasets/*/versions), else the training's own data.json."""
    d = os.path.join(workspace, ref)
    try:
        named = str((_read(os.path.join(d, 'data.json')) or {}).get('dataset_version') or '')
    except (OSError, ValueError):
        named = ''
    if named and named != 'none':
        import glob
        for v in sorted(glob.glob(os.path.join(workspace, 'datasets', '*', 'versions', 'v*.json'))):
            try:
                doc = _read(v)
            except (OSError, ValueError):
                continue
            if isinstance(doc, dict) and (doc.get('lineage') or {}).get('splits') and _version_label(doc) in named:
                return os.path.relpath(v, workspace)
    return f'{ref}/data.json'


def template(kind: str, workspace: str = '', title: str = '') -> dict:
    """A starting argument (module doc). Solutions point at the workspace's results (not bound: bind pins them)."""
    from .provenance import utc_now
    if kind not in TEMPLATES:
        raise GSNError(f'template: one of {", ".join(TEMPLATES)}')
    cand = evidence_candidates(workspace) if workspace else {k: [] for k in KINDS}
    nodes, links = [], []
    just = _n('J1', 'Justification', NOT_QUALIFIED)
    if kind == 'blank':
        nodes = [_n('G1', 'Goal', title or 'The system is acceptably safe in its operating domain', undeveloped=True),
                 just]
        links = [_l('G1', 'J1', 'InContextOf')]
    elif kind == 'iso26262':
        nodes = [
            _n('G1', 'Goal', 'The item is free of unreasonable risk due to hazards caused by malfunctioning behaviour '
                             'of E/E systems (ISO 26262)'),
            _n('C1', 'Context', 'The item definition and its operating domain (ISO 26262-3 §5)'),
            _n('C2', 'Context', 'The hazard analysis and risk assessment with its safety goals and ASILs (ISO 26262-3 '
                                '§6)'),
            just,
            _n('S1', 'Strategy', 'Argument over each safety goal of the HARA'),
            _n('G2', 'Goal', 'The hazards are identified and classified (HARA complete)', undeveloped=True),
            _n('G3', 'Goal', 'Each safety goal is met by the functional and technical safety concepts',
               undeveloped=True),
            _n('G4', 'Goal', 'The implementation is verified against the safety requirements'),
            _n('S2', 'Strategy', 'Argument over the verification evidence Falcon produces in simulation'),
            _n('A1', 'Assumption', 'The simulation is representative for the requirements verified here (to be shown '
                                   'by the validation of the simulation, outside Falcon)'),
            _n('G5', 'AwayGoal', 'Hardware and software faults are controlled (ISO 26262-5/-6)',
               module='hardware and software safety analyses')]
        links = [_l('G1', 'C1', 'InContextOf'), _l('G1', 'C2', 'InContextOf'), _l('G1', 'J1', 'InContextOf'),
                 _l('G1', 'S1'), _l('S1', 'G2'), _l('S1', 'G3'), _l('S1', 'G4'), _l('S1', 'G5'), _l('G4', 'S2'),
                 _l('S2', 'A1', 'InContextOf')]
        sols = [(k, r) for k in ('suite', 'ci', 'rules') for r in cand[k]]
        if not sols:
            nodes.append(_n('G6', 'Goal', 'Scenario suites pass their criteria', undeveloped=True))
            links.append(_l('S2', 'G6'))
        for i, (k, r) in enumerate(sols[:12], 1):
            claim = ({'measure': 'violations_per_km', 'limit': 1.0} if k == 'rules' else
                     {'measure': 'failure_rate', 'bound': 'clopper_pearson_upper', 'conf': 0.95, 'limit': 0.05})
            nodes.append(_n(f'Sn{i}', 'Solution', f'{k} result {r}', evidence={'kind': k, 'ref': r, 'claim': claim}))
            links.append(_l('S2', f'Sn{i}'))
    else:                                           # ISO 21448 SOTIF
        nodes = [
            _n('G1', 'Goal', 'The residual risk due to the intended functionality\'s insufficiencies is acceptable '
                             'in the operating domain (ISO 21448)'),
            _n('C1', 'Context', 'The operating domain (ODD) and the acceptance criteria for the residual risk (ISO '
                                '21448 clause 6)'),
            just,
            _n('S1', 'Strategy', 'Argument over the scenario areas of ISO 21448: known and unknown hazardous '
                                 'scenarios, and their triggering conditions'),
            _n('G2', 'Goal', 'Known hazardous scenarios (area 2) are evaluated and their residual risk is acceptable'),
            _n('G3', 'Goal', 'The unknown hazardous area (area 3) is reduced and its residual risk is bounded'),
            _n('G4', 'Goal', 'The triggering conditions are identified'),
            _n('A1', 'Assumption', 'The scenario parameters\' declared distributions are how they occur in the '
                                   'operating domain (core.failrate\'s nominal)')]
        links = [_l('G1', 'C1', 'InContextOf'), _l('G1', 'J1', 'InContextOf'), _l('G1', 'S1'), _l('S1', 'G2'),
                 _l('S1', 'G3'), _l('S1', 'G4'), _l('G3', 'A1', 'InContextOf')]
        k = 0

        def sol(parent, kind_, ref, claim, text):
            nonlocal k
            k += 1
            nodes.append(_n(f'Sn{k}', 'Solution', text, evidence={'kind': kind_, 'ref': ref, 'claim': claim}))
            links.append(_l(parent, f'Sn{k}'))
        for r in cand['suite'][:8]:
            sol('G2', 'suite', r, {'measure': 'failure_rate', 'bound': 'clopper_pearson_upper', 'conf': 0.95,
                                   'limit': 0.05}, f'suite {r}: its failure rate is bounded')
        for r in cand['ci'][:4]:
            sol('G2', 'ci', r, {'measure': 'failure_rate', 'bound': 'clopper_pearson_upper', 'conf': 0.95,
                                'limit': 0.05}, f'CI {r}: its failure rate is bounded')
        if workspace:
            try:
                from .oddcoverage import coverage
                cov = coverage(workspace)
                cnt = cov.get('counts') or {}
                nodes.append(_n('C2', 'Context', f'ODD coverage: {cov.get("runs", 0)} scenario runs over '
                                f'{(cov.get("domain") or {}).get("cells", 0)} cells — '
                                + ', '.join(f'{x.replace("_", " ")} {cnt[x]}' for x in cnt)))
                links.append(_l('G2', 'C2', 'InContextOf'))
            except Exception:                    # noqa: BLE001 (no coverage to state: the template goes without)
                pass
        def ran(r):
            w = where_text(where_ran(os.path.join(workspace, r))) if workspace else ''
            return f' ({w})' if w and w != 'on this machine' else ''
        for r in cand['falsify'][:8]:
            sol('G3', 'falsify', r, {'measure': 'min_robustness', 'limit': 0.0},
                f'falsification {r}{ran(r)}: the requirement was not falsified')
        for r in cand['failrate'][:8]:
            sol('G3', 'failrate', r, {'measure': 'failure_rate', 'bound': 'ci95_upper', 'limit': 1e-3},
                f'failure rate {r}{ran(r)}: its 95 % upper bound is below the limit')
        if not (cand['suite'] or cand['ci']):
            nodes.append(_n('G5', 'Goal', 'Scenario suites cover the known hazardous scenarios', undeveloped=True))
            links.append(_l('G2', 'G5'))
        if not (cand['falsify'] or cand['failrate']):
            nodes.append(_n('G6', 'Goal', 'Falsification and failure-rate studies search the unknown area',
                            undeveloped=True))
            links.append(_l('G3', 'G6'))
        tcs = []
        if workspace:
            try:
                from .safetycase import triggering_conditions
                sd = os.path.join(workspace, 'studies')
                studies = [os.path.join(sd, s) for s in sorted(os.listdir(sd))] if os.path.isdir(sd) else []
                tcs = [t for t in triggering_conditions({}, studies) if t.get('failures')]
            except Exception:                    # noqa: BLE001
                tcs = []
        for i, t in enumerate(tcs[:10], 1):
            span = '; '.join(f'{p} {v["failing"][0]:.3g}–{v["failing"][1]:.3g} (tried {v["tried"][0]:.3g}–'
                             f'{v["tried"][1]:.3g})' for p, v in (t.get('parameters') or {}).items()
                             if v.get('failing'))
            nodes.append(_n(f'C{10 + i}', 'Context', f'Triggering condition, {t["source"]} ({t.get("scenario")}): '
                            f'{t["failures"]} failures of {t.get("evaluations")}' + (f' — {span}' if span else '')))
            links.append(_l('G4', f'C{10 + i}', 'InContextOf'))
        studies_with = [r for r in cand['falsify']]
        if studies_with:
            sol('G4', 'falsify', studies_with[0], None,
                f'falsification {studies_with[0]}: its falsified runs (pinned) are the triggering conditions found')
        else:
            n4 = node({'nodes': nodes}, 'G4')
            n4['undeveloped'] = True
        if cand['lineage']:                          # machine-learned functions: each model's data, by its root
            nodes += [_n('G7', 'Goal', 'The machine-learned models are trained on the data stated for them, and '
                                       'neither changed since'),
                      _n('C3', 'Context', 'Each model\'s ONNX file names its training data by an RFC 6962 Merkle root '
                                          '(falcon.data_merkle_root; core.lineage)')]
            links += [_l('S1', 'G7'), _l('G7', 'C3', 'InContextOf')]
            for r in cand['lineage'][:8]:
                data = model_data(workspace, r)
                sol('G7', 'lineage', r, {'measure': 'data_root', 'data': data},
                    f'model {r}: its data_merkle_root is the root of {data}')
    return {'schema': SCHEMA, 'title': title or {'iso26262': 'Functional safety (ISO 26262)',
                                                  'sotif': 'Safety of the intended functionality (ISO 21448)',
                                                  'blank': 'Safety case'}[kind],
            'template': kind, 'created': utc_now(), 'nodes': nodes, 'links': links,
            'notes': [NOT_QUALIFIED]}


# ================================================================================================ drawing
BOX_W, BOX_H, GAP_X, GAP_Y = 210.0, 74.0, 26.0, 56.0


def layout(doc: dict) -> dict:
    """{id: (x, y, w, h)} — a tidy tree from the top goals down both link kinds (a node with two parents is placed
    under the first); leaves left to right, a parent centred over its children; nodes nothing reaches go last."""
    kids = children(doc)
    order = top_goals(doc) + [n['id'] for n in doc['nodes']]
    pos, depth = {}, {}
    nxt = [0.0]

    def place(nid, d):
        if nid in depth:
            return
        depth[nid] = d
        cs = [c for c in kids.get(nid, []) if c not in depth]
        for c in cs:
            place(c, d + 1)
        mine = [c for c in cs if depth.get(c) == d + 1 and c in pos]
        if mine:
            x = (min(pos[c][0] for c in mine) + max(pos[c][0] for c in mine)) / 2.0
        else:
            x = nxt[0]
            nxt[0] += BOX_W + GAP_X
        pos[nid] = (x, d * (BOX_H + GAP_Y))
    for nid in order:
        if nid not in depth:
            place(nid, 0)
    return {k: (x, y, BOX_W, BOX_H) for k, (x, y) in pos.items()}


STATUS_FILL = {'green': '#2fbf71', 'amber': '#f2a93b', 'red': '#ef5350', 'grey': '#9a9993', None: '#d9d7d0'}


def _wrap(text: str, width: int = 34, lines: int = 4) -> list:
    words, out, cur = (text or '').split(), [], ''
    for w in words:
        if len(cur) + len(w) + 1 > width and cur:
            out.append(cur)
            cur = w
        else:
            cur = f'{cur} {w}'.strip()
    if cur:
        out.append(cur)
    if len(out) > lines:
        out = out[:lines]
        out[-1] = out[-1][:width - 1] + '…'
    return out


def svg(doc: dict, v: dict | None = None) -> str:
    """The argument as a standalone SVG (GSN shapes; fill by status; SupportedBy solid with a filled head,
    InContextOf dashed with a hollow head; every node has a <title> with its status and why)."""
    from html import escape
    lay = layout(doc)
    by = {n['id']: n for n in doc['nodes']}
    st = (v or {}).get('nodes') or {}
    if not lay:
        return '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"/>'
    W = max(x + w for x, _, w, _ in lay.values()) + 20
    H = max(y + h for _, y, _, h in lay.values()) + 30
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="-10 -10 {W:.0f} {H:.0f}" width="{W:.0f}" '
           f'height="{H:.0f}" font-family="IBM Plex Sans, system-ui, sans-serif" font-size="11">',
           '<defs><marker id="sb" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="8" markerHeight="8" '
           'orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#52514e"/></marker>'
           '<marker id="ic" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="8" markerHeight="8" orient="auto">'
           '<path d="M0,0 L10,5 L0,10 z" fill="#ffffff" stroke="#52514e"/></marker></defs>']
    for ln in doc['links']:
        if ln['from'] not in lay or ln['to'] not in lay:
            continue
        x0, y0, w0, h0 = lay[ln['from']]
        x1, y1, w1, _ = lay[ln['to']]
        dash = ' stroke-dasharray="5 4"' if ln['kind'] == 'InContextOf' else ''
        mk = 'ic' if ln['kind'] == 'InContextOf' else 'sb'
        out.append(f'<line x1="{x0 + w0 / 2:.1f}" y1="{y0 + h0:.1f}" x2="{x1 + w1 / 2:.1f}" y2="{y1:.1f}" '
                   f'stroke="#52514e" stroke-width="1.2"{dash} marker-end="url(#{mk})"/>')
    for nid, (x, y, w, h) in lay.items():
        n = by[nid]
        s = (st.get(nid) or {}).get('status')
        fill = STATUS_FILL.get(s, STATUS_FILL[None]) if n['type'] not in CONTEXTS else '#f3f2ef'
        t = n['type']
        attrs = f'fill="{fill}" fill-opacity="0.35" stroke="{STATUS_FILL.get(s, "#52514e") if s else "#52514e"}" ' \
                'stroke-width="1.6"'
        if t == 'Strategy':
            shape = f'<polygon points="{x + 14},{y} {x + w},{y} {x + w - 14},{y + h} {x},{y + h}" {attrs}/>'
        elif t == 'Solution':
            shape = f'<ellipse cx="{x + w / 2}" cy="{y + h / 2}" rx="{w / 2}" ry="{h / 2}" {attrs}/>'
        elif t == 'Context':
            shape = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="22" {attrs}/>'
        elif t in ('Assumption', 'Justification'):
            shape = (f'<ellipse cx="{x + w / 2}" cy="{y + h / 2}" rx="{w / 2}" ry="{h / 2}" {attrs}/>'
                     f'<text x="{x + w - 6}" y="{y + h - 4}" font-weight="600">{t[0]}</text>')
        elif t == 'AwayGoal':
            shape = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" {attrs}/>'
                     f'<line x1="{x}" y1="{y + h - 16}" x2="{x + w}" y2="{y + h - 16}" stroke="#52514e"/>')
        else:
            shape = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" {attrs}/>'
        if n.get('undeveloped') or (t in SUPPORTERS and not children(doc, 'SupportedBy').get(nid)):
            cx, cy = x + w / 2, y + h
            shape += (f'<polygon points="{cx},{cy} {cx + 8},{cy + 8} {cx},{cy + 16} {cx - 8},{cy + 8}" '
                      'fill="#ffffff" stroke="#52514e"/>')
        why = (st.get(nid) or {}).get('why') or ''
        lines = [f'<text x="{x + 8}" y="{y + 15}" font-weight="600">{escape(nid)}'
                 + (f' · {escape(s)}' if s else '') + '</text>']
        for i, ln_ in enumerate(_wrap(n.get('text', ''))):
            lines.append(f'<text x="{x + 8}" y="{y + 30 + 12 * i}">{escape(ln_)}</text>')
        out.append(f'<g><title>{escape(f"{nid} ({t}): {s or "no status"} — {why}")}</title>{shape}'
                   + ''.join(lines) + '</g>')
    out.append('</svg>')
    return '\n'.join(out)


def report(src: str):
    """The 'gsn' report (core.reports): src a workspace's safety folder, gsn.json or the workspace."""
    from .reports import Page, esc, kv, table
    p = src
    if os.path.isdir(p) and os.path.isfile(os.path.join(p, 'safety', 'gsn.json')):
        p = os.path.join(p, 'safety', 'gsn.json')
    elif os.path.isdir(p):
        p = os.path.join(p, 'gsn.json')
    ws = os.path.dirname(os.path.dirname(os.path.abspath(p)))
    pg = Page('gsn', 'Safety case (GSN)', p)
    doc = pg.json(p)
    probs = validate(doc)
    if probs:
        raise GSNError('; '.join(probs[:5]))
    v = verify(ws, doc)
    pg.title = doc.get('title') or pg.title
    pg.add(kv([('template', esc(doc.get('template'))), ('top goals', esc(', '.join(f'{g}: {s}' for g, s in
                                                                                    v['top'].items()))),
               ('nodes', esc(', '.join(f'{k} {n}' for k, n in v['counts'].items()))), ('verified', esc(v['at']))]))
    pg.note(NOT_QUALIFIED)
    pg.h2('The argument')
    pg.add(svg(doc, v))
    pg.h2('Evidence')
    rows = []
    for n in doc['nodes']:
        if n['type'] != 'Solution':
            continue
        r = v['nodes'][n['id']]
        ev = n.get('evidence') or {}
        cls = {'green': 'good', 'red': 'crit', 'amber': 'warn'}.get(r['status'], 'muted')
        rows.append([esc(n['id']), f'<span class="{cls}">{esc(r["status"])}</span>', esc(ev.get('kind')),
                     f'<span class="mono">{esc(ev.get("ref"))}</span>'
                     + (f'<br>{esc(where_text(ev.get("where")))}' if ev.get('where') else ''),
                     esc(json.dumps(ev.get('claim'))),
                     f'<span class="mono">{esc((ev.get("sha256") or "")[:16])}</span>', esc(r['why'])])
    pg.add(table(['node', 'status', 'kind', 'result', 'claim', 'pinned SHA-256', 'why'], rows))
    pg.h2('Nodes')
    pg.add(table(['node', 'type', 'text', 'status', 'why'],
                 [[esc(n['id']), esc(n['type']), esc(n.get('text')), esc(v['nodes'][n['id']]['status'] or '—'),
                   esc(v['nodes'][n['id']]['why'])] for n in doc['nodes']]))
    evs = events(ws, 30)
    if evs:
        pg.h2('Recent events')
        pg.add(table(['at', 'event', 'by', 'detail'], [[esc(e.get('at')), esc(e.get('event')), esc(e.get('by')),
                                                       esc(json.dumps({k: x for k, x in e.items() if k not in (
                                                           'schema', 'at', 'event', 'by', 'statuses')},
                                                           default=str)[:300])] for e in reversed(evs)]))
    pg.provenance(None, 'the safety case')
    return pg


def export(workspace: str, out: str, by: str = 'cli') -> str:
    """The safety case as a self-contained HTML report (core.reports kind gsn) or a standalone SVG, by out's
    extension; logged."""
    if out.endswith('.svg'):
        from .paths import atomic_write
        doc = load(workspace)
        atomic_write(out, svg(doc, verify(workspace, doc)) + '\n')
    else:
        from .reports import write_report
        write_report('gsn', gsn_path(workspace), out)
    log_event(workspace, 'export', by, path=os.path.abspath(out))
    return os.path.abspath(out)


def tree_text(doc: dict, v: dict) -> str:
    by = {n['id']: n for n in doc['nodes']}
    lines, seen = [], set()

    def walk(nid, depth, kind):
        n = by[nid]
        r = v['nodes'].get(nid) or {}
        mark = '' if kind == 'SupportedBy' else '(in context) '
        lines.append(f'{"  " * depth}{mark}{nid} [{n["type"]}] {r.get("status") or ""}: {n.get("text", "")[:90]}'
                     + (f'  — {r.get("why")}' if r.get('status') else ''))
        if nid in seen:
            return
        seen.add(nid)
        for ln in doc['links']:
            if ln['from'] == nid:
                walk(ln['to'], depth + 1, ln['kind'])
    for g in top_goals(doc):
        walk(g, 0, 'SupportedBy')
    return '\n'.join(lines)


# ================================================================================================ the command
def _workspace(a) -> str:
    if a.workspace:
        return os.path.abspath(a.workspace)
    if a.project:
        from .project import Project
        return Project.load(a.project).workspace
    raise GSNError('give --project PROJECT (or --workspace DIR)')


def main_gsn(argv) -> int:
    """`falcon gsn new|show|bind|verify|export …` (module doc): 0 ok, 1 a top goal red (verify --gate), 2 failed."""
    import argparse
    from .project import ProjectError
    ap = argparse.ArgumentParser(prog='falcon gsn', description='the living GSN safety case (docs/gsn.md)')
    sub = ap.add_subparsers(dest='action', required=True)

    def common(p):
        p.add_argument('--project', default='', help='the project (its workspace holds safety/gsn.json)')
        p.add_argument('--workspace', default='', help='or the workspace folder itself')
    n = sub.add_parser('new', help='start a safety case from a template')
    common(n)
    n.add_argument('--template', default='sotif', choices=TEMPLATES)
    n.add_argument('--title', default='')
    n.add_argument('--bind', action='store_true', help='bind every Solution the template points at a result')
    n.add_argument('--force', action='store_true', help='replace an existing safety case')
    s = sub.add_parser('show', help='the argument as a tree with each node\'s status')
    common(s)
    s.add_argument('--json', action='store_true')
    b = sub.add_parser('bind', help='bind (or re-bind) evidence to a Solution, pinning its hash')
    common(b)
    b.add_argument('node')
    b.add_argument('--kind', choices=KINDS, default='')
    b.add_argument('--ref', default='', help='the result: a path in the workspace, or ci:NAME@latest')
    b.add_argument('--measure', choices=MEASURES, default='')
    b.add_argument('--bound', choices=BOUNDS, default=None)
    b.add_argument('--conf', type=float, default=0.95)
    b.add_argument('--limit', type=float, default=None)
    b.add_argument('--data', default='', help='data_root: the data the model must name (a dataset version\'s JSON '
                                              'or a data.json; default the training\'s own)')
    b.add_argument('--claim', default='', help='the claim as JSON (instead of --measure …)')
    b.add_argument('--unbind', action='store_true', help='remove the evidence (the Solution becomes undeveloped)')
    v = sub.add_parser('verify', help='re-verify every node; --gate exits 1 when a top goal is red')
    common(v)
    v.add_argument('--json', action='store_true')
    v.add_argument('--gate', action='store_true')
    v.add_argument('--fail-on', default='red', choices=('red', 'amber'))
    e = sub.add_parser('export', help='the safety case as an HTML report or an SVG')
    common(e)
    e.add_argument('--out', required=True, help='FILE.html or FILE.svg')
    try:
        a = ap.parse_args(argv)
    except SystemExit as ex:
        return 0 if ex.code == 0 else 2
    try:
        ws = _workspace(a)
        if a.action == 'new':
            if os.path.isfile(gsn_path(ws)) and not a.force:
                raise GSNError(f'{gsn_path(ws)} exists: give --force to replace it')
            doc = template(a.template, ws, a.title)
            bound = []
            if a.bind:
                for nd in doc['nodes']:
                    ev = nd.get('evidence')
                    if ev:
                        try:
                            bind(ws, doc, nd['id'], ev['kind'], ev['ref'], ev.get('claim'))
                            bound.append(nd['id'])
                        except GSNError as ex:
                            print(f'not bound: {nd["id"]}: {ex}', file=sys.stderr)
            save(ws, doc)
            log_event(ws, 'new', 'cli', template=a.template, nodes=len(doc['nodes']), bound=bound,
                      gsn_sha256=doc_sha(doc))
            print(f'{gsn_path(ws)}: {len(doc["nodes"])} nodes ({a.template})' + (f', {len(bound)} bound'
                                                                                 if a.bind else ''))
            return 0
        doc = load(ws)
        if a.action == 'bind':
            nd = node(doc, a.node)
            old = copy.deepcopy(nd.get('evidence'))
            if a.unbind:
                nd.pop('evidence', None)
                save(ws, doc)
                log_event(ws, 'unbind', 'cli', node=a.node, previous=old, gsn_sha256=doc_sha(doc))
                print(f'{a.node}: evidence removed')
                return 0
            claim = json.loads(a.claim) if a.claim else (
                {k: x for k, x in {'measure': a.measure, 'bound': a.bound, 'conf': a.conf, 'limit': a.limit,
                                   'data': a.data}.items() if x not in (None, '')} if a.measure else None)
            if claim and claim.get('measure') == 'data_root':
                claim = {k: x for k, x in claim.items() if k in ('measure', 'data')}
            kind = a.kind or (old or {}).get('kind')
            ref = a.ref or (old or {}).get('ref')
            if not kind or not ref:
                raise GSNError('give --kind and --ref (or re-bind a Solution that has them)')
            ev = bind(ws, doc, a.node, kind, ref, claim, keep_claim=not a.claim and not a.measure)
            save(ws, doc)
            log_event(ws, 'bind', 'cli', node=a.node, evidence=ev, previous=old, gsn_sha256=doc_sha(doc))
            print(f'{a.node}: bound to {ev["path"]} ({ev["sha256"][:12]})')
            return 0
        if a.action == 'export':
            print(export(ws, a.out))
            return 0
        res = verify_and_log(ws, 'cli', doc=doc) if a.action == 'verify' else verify(ws, doc)
        if a.json:
            print(json.dumps(res if a.action == 'verify' else {'gsn': doc, 'verify': res}, indent=2, default=str))
        else:
            print(tree_text(doc, res))
            print('top goals: ' + ', '.join(f'{g} {s}' for g, s in res['top'].items()))
        if a.action == 'verify' and a.gate:
            return 0 if gate(res, fail_on=a.fail_on)[0] else 1
        return 0
    except (GSNError, ProjectError, OSError, ValueError) as ex:
        print(f'failed: {ex}', flush=True)
        return 2
