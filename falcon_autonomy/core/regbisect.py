"""Regression bisect (standard library; CI mode inside the functions): given two versions of a driving stack — two git
refs of the stack's repository — find which scenarios got worse, then the first commit that broke each one.
`falcon bisect run`; docs/bisect.md is the manual.

1. The endpoints. The stack's CI config (core.ci, falcon.ci/1: its suite of scenarios × seeds) runs on the good and the
   bad commit, `repeats` times each when the mode is not deterministic. Per scenario, its finished runs give k passed
   of n at each end. Which scenarios regressed (find_regressions):
     deterministic mode (kinematic motion, no traffic, a stack not declared otherwise: a run repeats bit for bit) —
       a scenario regressed when at least one of its runs (the same seed and parameter values: CI mode's pairing key)
       passed at the good end and fails at the bad end. Repeats would repeat the same outcome, so there is one run.
     otherwise — Fisher's exact test, one-sided, on the 2 × 2 table (passed, failed) × (good, bad): the scenario
       regressed when P(a pass count at the bad end this low or lower | both ends' pass counts) < alpha (default
       0.05) and its pass rate fell. With n runs per end the smallest p reachable is 1 / C(2n, n): 0.5 for one run,
       0.167 for two, 0.05 for three, 0.014 for four — fewer than four runs per end and scenario can never show a
       regression at 0.05, and the output says so.
2. The bisection (Bisector, pure: unit-tested with a fake evaluator). The commits from the good ref (exclusive) to the
   bad ref (inclusive) along the first-parent chain (`git rev-list --first-parent --ancestry-path`: a merge is one
   step, as `git bisect --first-parent`). Each regressed scenario keeps its own interval (last good, first bad);
   every step tests the commit nearest the middle of the widest interval, and one commit's runs answer for every
   scenario whose interval holds it. A commit is
     skipped for every scenario — like `git bisect skip` — when its checkout or the build command fails (`--build`;
       exit 125 or any failure), or its CI run ends without a finished run of that scenario;
     bad for a scenario — deterministic: when any run that regressed between the endpoints (its pairing key) fails
       there; otherwise: when its pass rate is nearer the bad end's than the good end's (below their midpoint; at
       the midpoint exactly it is undecided and counts as skipped);
     good otherwise.
   The first bad commit is found when the interval closes; when only skipped commits are left between the last good
   and the first bad, the answer is the list of them (git's "the first bad commit could be any of"). A good verdict
   at a commit after a known-bad one (a fix, then a break again; or noise) is noted: bisection assumes one change of
   state between the ends.
3. Each commit is checked out with `git worktree add --detach` into OUT/checkouts/<sha> (removed afterwards unless
   --keep-checkouts), the config is read from that checkout (so the suite and the stack's own launch files are the
   commit's), `{repo}` in the stack's commands becomes the checkout's path, and CI mode runs it into
   OUT/commits/<sha>/run-<k> with no baseline and no store. A commit already evaluated in OUT is read back, not run
   again (a bisect resumes).

Outputs: OUT/bisect.json (BISECT_SCHEMA: the endpoints, the commits, the test and its rule, each scenario's verdict
and first bad commit with its author, date and subject, every step), OUT/summary.md.

Limits: one change of state per scenario between the ends is assumed (a flaky scenario in a deterministic mode, or a
break, fix and break again, can mislead it — both are noted when seen); the midpoint rule on noisy scenarios can
misclassify a commit with few runs (more repeats, fewer mistakes: each commit's pass count is in the output); every
commit costs a full CI run of the regressed scenarios' suite.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import time

BISECT_SCHEMA = 'falcon.bisect/1'
GOOD, BAD, SKIP = 'good', 'bad', 'skip'
ALPHA = 0.05
MAX_REPEATS = 20


class BisectError(Exception):
    pass


# ============================================================================================== the statistics
def fisher_less(k_good: int, n_good: int, k_bad: int, n_bad: int) -> float:
    """One-sided Fisher exact test: P(X ≤ k_bad) for X the bad end's pass count under the hypergeometric law given
    the margins (both ends' runs and the total passes) — small when the bad end passes less than chance allows."""
    N, K = n_good + n_bad, k_good + k_bad
    if n_bad <= 0 or n_good <= 0:
        return 1.0
    den = math.comb(N, n_bad)
    lo = max(0, n_bad - (N - K))
    return min(1.0, sum(math.comb(K, x) * math.comb(N - K, n_bad - x) for x in range(lo, k_bad + 1)) / den)


def min_p(n_good: int, n_bad: int) -> float:
    """The smallest one-sided p the test can give with these run counts (all pass at one end, all fail at the other)."""
    return fisher_less(n_good, n_good, 0, n_bad) if n_good and n_bad else 1.0


def tally(records: list) -> dict:
    """scenario -> {'k': passed, 'n': finished, 'pairs': {pairing key: [passed, …]}} over CI run records (core.ci
    RUN_SCHEMA: group, pair, status, passed)."""
    out: dict = {}
    for r in records:
        if r.get('status') != 'ok' or r.get('passed') is None:
            continue
        t = out.setdefault(r.get('group') or '?', {'k': 0, 'n': 0, 'pairs': {}, 'names': {}})
        t['n'] += 1
        t['k'] += 1 if r['passed'] else 0
        t['pairs'].setdefault(r.get('pair') or r.get('name'), []).append(bool(r['passed']))
        t['names'][r.get('pair') or r.get('name')] = r.get('name')
    return out


def find_regressions(good: dict, bad: dict, deterministic: bool, alpha: float = ALPHA) -> list:
    """Per scenario seen at both ends: the counts, the test, and whether it regressed (module doc). good, bad: tally()
    of each end."""
    rows = []
    for s in sorted(set(good) | set(bad)):
        g, b = good.get(s), bad.get(s)
        row = {'scenario': s, 'good': {'k': g['k'], 'n': g['n']} if g else None,
               'bad': {'k': b['k'], 'n': b['n']} if b else None, 'regressed': False, 'improved': False}
        if not g or not b:
            row['why'] = 'not run at both ends' if (g or b) else 'not run'
            rows.append(row)
            continue
        if deterministic:
            flipped = sorted(p for p, v in b['pairs'].items() if p in g['pairs'] and all(g['pairs'][p]) and
                             not any(v))
            fixed = sorted(p for p, v in b['pairs'].items() if p in g['pairs'] and not any(g['pairs'][p]) and all(v))
            row.update(test='paired, deterministic', flipped=flipped,
                       flipped_names=[b['names'].get(p, p) for p in flipped], regressed=bool(flipped),
                       improved=bool(fixed) and not flipped,
                       why=(f'{len(flipped)} run(s) passed at the good end and fail at the bad end' if flipped else
                            'no run that passed at the good end fails at the bad end'))
        else:
            p = fisher_less(g['k'], g['n'], b['k'], b['n'])
            pi = fisher_less(b['k'], b['n'], g['k'], g['n'])
            fell = b['k'] / b['n'] < g['k'] / g['n']
            row.update(test='Fisher exact, one-sided', p=round(p, 6), alpha=alpha, regressed=bool(fell and p < alpha),
                       improved=bool(not fell and pi < alpha and b['k'] / b['n'] > g['k'] / g['n']),
                       min_p=round(min_p(g['n'], b['n']), 6))
            if row['min_p'] >= alpha:
                row['why'] = (f'{g["n"]} and {b["n"]} runs cannot show a change at {alpha:g} (the smallest p is '
                              f'{row["min_p"]:.3g}): more repeats')
            else:
                row['why'] = (f'pass rate {g["k"]}/{g["n"]} → {b["k"]}/{b["n"]}, p = {p:.3g}' +
                              (' < ' if p < alpha else ' ≥ ') + f'{alpha:g}')
        rows.append(row)
    return rows


def classify(t: dict | None, reg: dict, deterministic: bool) -> str:
    """A commit's verdict for one regressed scenario from its runs' tally (module doc)."""
    if not t or not t['n']:
        return SKIP
    if deterministic:
        seen = [p for p in reg.get('flipped') or () if p in t['pairs']]
        if not seen:
            return SKIP
        return BAD if any(not all(t['pairs'][p]) for p in seen) else GOOD
    pg = reg['good']['k'] / reg['good']['n']
    pb = reg['bad']['k'] / reg['bad']['n']
    pc = t['k'] / t['n']
    mid = (pg + pb) / 2.0
    if abs(pc - mid) < 1e-12:
        return SKIP
    return BAD if pc < mid else GOOD


# ============================================================================================== the bisection
class Bisector:
    """Bisect n commits (index 0 … n-1: the commits after the good ref, the last being the bad ref, known bad) for
    several scenarios at once (module doc). next() is the index to test (None: done), record(i, {scenario:
    verdict}) what it said; result() each scenario's answer."""

    def __init__(self, n: int, scenarios):
        if n < 1:
            raise BisectError('no commits between the good and the bad ref')
        self.n = int(n)
        self.scenarios = list(scenarios)
        self.lo = {s: -1 for s in self.scenarios}            # the last known good index (-1: the good ref)
        self.hi = {s: self.n - 1 for s in self.scenarios}    # the first known bad index
        self.skipped = {s: set() for s in self.scenarios}
        self.tested: dict = {}                               # index -> {scenario: verdict}
        self.notes: list = []

    def candidates(self, s) -> list:
        return [i for i in range(self.lo[s] + 1, self.hi[s]) if i not in self.skipped[s] and i not in self.tested]

    def open(self) -> list:
        return [s for s in self.scenarios if self.candidates(s)]

    def next(self):
        best = None
        for s in self.scenarios:
            c = self.candidates(s)
            if not c:
                continue
            mid = (self.lo[s] + self.hi[s]) / 2.0
            pick = min(c, key=lambda i: (abs(i - mid), i))
            key = (-(self.hi[s] - self.lo[s]), pick)
            if best is None or key < best[0]:
                best = (key, pick)
        return None if best is None else best[1]

    def record(self, i: int, verdicts: dict):
        self.tested[i] = dict(verdicts)
        for s in self.scenarios:
            v = verdicts.get(s, SKIP)
            if v == SKIP:
                self.skipped[s].add(i)
            elif self.lo[s] < i < self.hi[s]:
                if v == GOOD:
                    self.lo[s] = i
                else:
                    self.hi[s] = i
            elif v == GOOD and i >= self.hi[s]:
                self.notes.append(f'{s}: commit {i} is good after the bad commit {self.hi[s]} (a fix and a break '
                                  'again, or a flaky run): bisection assumes one change')
            elif v == BAD and i <= self.lo[s]:
                self.notes.append(f'{s}: commit {i} is bad before the good commit {self.lo[s]} (a flaky run, or a '
                                  'break fixed and broken again): bisection assumes one change')

    def result(self) -> dict:
        out = {}
        for s in self.scenarios:
            lo, hi = self.lo[s], self.hi[s]
            between = list(range(lo + 1, hi))
            left = self.candidates(s)
            if not between:
                out[s] = {'status': 'found', 'first_bad': hi, 'last_good': lo, 'candidates': [hi]}
            elif not left:
                out[s] = {'status': 'ambiguous', 'first_bad': None, 'last_good': lo,
                          'candidates': between + [hi], 'why': 'only skipped commits are left between the last good '
                                                               'and the first bad commit'}
            else:
                out[s] = {'status': 'stopped', 'first_bad': None, 'last_good': lo, 'candidates': between + [hi],
                          'why': f'{len(left)} commit(s) still untested (the step limit was reached)'}
            out[s]['tested'] = sum(1 for i, v in self.tested.items() if v.get(s, SKIP) != SKIP)
        return out


def run_bisect(n: int, scenarios, evaluate, max_steps: int | None = None, on_step=None) -> dict:
    """Bisect with evaluate(index) -> {scenario: verdict}; returns {'result': Bisector.result(), 'steps': [...],
    'notes': [...]}. on_step(step dict) after each evaluation."""
    b = Bisector(n, scenarios)
    steps = []
    while True:
        i = b.next()
        if i is None or (max_steps is not None and len(steps) >= max_steps):
            break
        v = evaluate(i)
        b.record(i, v)
        steps.append({'index': i, 'verdicts': dict(v)})
        if on_step:
            on_step(steps[-1])
    return {'result': b.result(), 'steps': steps, 'notes': b.notes, 'bisector': b}


# ============================================================================================== git
def git(repo: str, *args, check=True, timeout=120) -> str:
    try:
        p = subprocess.run(['git', '-C', repo, *args], capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise BisectError(f'git {" ".join(args[:2])}: {e}') from e
    if check and p.returncode != 0:
        raise BisectError(f'git {" ".join(args[:3])}: {(p.stderr or p.stdout).strip()[:300]}')
    return p.stdout.strip()


def resolve_ref(repo: str, ref: str) -> str:
    return git(repo, 'rev-parse', '--verify', f'{ref}^{{commit}}')


def commit_info(repo: str, sha: str) -> dict:
    raw = git(repo, 'log', '-1', '--format=%H%x1f%an%x1f%aI%x1f%s', sha)
    h, an, ad, subj = (raw.split('\x1f') + ['', '', '', ''])[:4]
    return {'sha': h, 'short': h[:10], 'author': an, 'date': ad, 'subject': subj}


def commits_between(repo: str, good: str, bad: str) -> list:
    """The commits after good up to and including bad on bad's first-parent chain (oldest first)."""
    p = subprocess.run(['git', '-C', repo, 'merge-base', '--is-ancestor', good, bad], capture_output=True)
    if p.returncode != 0:
        raise BisectError('the good ref is not an ancestor of the bad ref: bisect needs good older than bad on one '
                          'history')
    shas = git(repo, 'rev-list', '--first-parent', '--ancestry-path', '--reverse', f'{good}..{bad}').split()
    if not shas:
        raise BisectError('the good and the bad ref are the same commit')
    return [commit_info(repo, s) for s in shas]


# ============================================================================================== one commit, by CI mode
class CIEvaluator:
    """Runs a commit's CI config (module doc, 3) and caches its records in OUT/commits/<sha>/records.json."""

    def __init__(self, repo: str, config: str, out: str, repeats: int = 1, overrides: dict | None = None,
                 build: str = '', keep_checkouts: bool = False, log=None, cancel=None):
        self.repo = os.path.realpath(repo)
        self.config = config                     # relative to the repository's root
        self.out = os.path.abspath(out)
        self.repeats = max(1, int(repeats))
        self.overrides = dict(overrides or {})
        self.build = build
        self.keep = keep_checkouts
        self.log = log or (lambda t: None)
        self.cancel = cancel

    def _cache(self, sha):
        return os.path.join(self.out, 'commits', sha[:12], 'records.json')

    def records(self, sha: str) -> dict:
        """{'status': 'ok'|'skip', 'why', 'records': [...], 'deterministic', 'ci': [output folders]}."""
        from .provenance import read_json, write_json
        cp = self._cache(sha)
        if os.path.isfile(cp):
            try:
                doc = read_json(cp)
                if doc.get('repeats') == self.repeats:
                    return doc
            except (OSError, ValueError):
                pass
        doc = self._run(sha)
        doc['repeats'] = self.repeats
        write_json(cp, doc)
        return doc

    def _run(self, sha: str) -> dict:
        from . import ci as CI
        wt = os.path.join(self.out, 'checkouts', sha[:12])
        if os.path.isdir(wt):
            git(self.repo, 'worktree', 'remove', '--force', wt, check=False)
            shutil.rmtree(wt, ignore_errors=True)
        try:
            git(self.repo, 'worktree', 'add', '--detach', '--force', wt, sha)
        except BisectError as e:
            return {'status': SKIP, 'why': f'checkout failed: {e}', 'records': [], 'ci': []}
        try:
            if self.build:
                self.log(f'{sha[:10]}: build: {self.build}')
                try:
                    p = subprocess.run(self.build, shell=True, cwd=wt, capture_output=True, text=True, timeout=3600,
                                       env=dict(os.environ, FALCON_BISECT_COMMIT=sha))
                except subprocess.TimeoutExpired:
                    return {'status': SKIP, 'why': 'the build timed out (1 h)', 'records': [], 'ci': []}
                if p.returncode != 0:
                    tail = (p.stderr or p.stdout or '').strip().splitlines()[-3:]
                    return {'status': SKIP, 'why': f'the build failed (exit {p.returncode}' +
                            (', 125: skip' if p.returncode == 125 else '') + ')' +
                            (': ' + ' | '.join(tail) if tail else ''), 'records': [], 'ci': []}
            cfg_path = os.path.join(wt, self.config)
            if not os.path.isfile(cfg_path):
                return {'status': SKIP, 'why': f'{self.config} does not exist in this commit', 'records': [],
                        'ci': []}
            derived = derive_config(cfg_path, wt, sha)
            recs, dirs, det, why = [], [], None, ''
            for k in range(self.repeats):
                if self.cancel is not None and self.cancel():
                    raise KeyboardInterrupt
                out = os.path.join(self.out, 'commits', sha[:12], f'run-{k}')
                o = dict(self.overrides, out=out, no_baseline=True, force=True)
                self.log(f'{sha[:10]}: CI run {k + 1} of {self.repeats}')
                try:
                    res, code = CI.run_ci(derived, o, log=lambda t: None, env={})
                except CI.CIError as e:
                    return {'status': SKIP, 'why': f'CI mode refused the config: {e}', 'records': [], 'ci': dirs}
                dirs.append(out)
                if res.get('verdict') == 'cancelled':
                    raise KeyboardInterrupt
                det = res.get('deterministic')
                why = res.get('error') or why
                for r in res.get('runs') or ():
                    recs.append({k2: r.get(k2) for k2 in ('group', 'pair', 'name', 'seed', 'status', 'passed',
                                                         'values')} | {'repeat': k})
            st = 'ok' if any(r['status'] == 'ok' for r in recs) else SKIP
            return {'status': st, 'why': why if st == SKIP else '', 'records': recs, 'ci': dirs,
                    'deterministic': det}
        finally:
            if not self.keep:
                git(self.repo, 'worktree', 'remove', '--force', wt, check=False)
                shutil.rmtree(wt, ignore_errors=True)
                git(self.repo, 'worktree', 'prune', check=False)


def derive_config(cfg_path: str, checkout: str, sha: str) -> str:
    """The commit's CI config with `{repo}` in the stack's commands set to its checkout, the subject's repository the
    checkout, and no baseline or store: written beside it (relative paths keep meaning the same)."""
    from . import ci as CI
    raw = CI.read_doc(cfg_path)

    def sub(v):
        if isinstance(v, str):
            return v.replace('{repo}', checkout)
        if isinstance(v, list):
            return [sub(x) for x in v]
        if isinstance(v, dict):
            return {k: sub(x) for k, x in v.items()}
        return v
    raw = sub(raw)
    raw.setdefault('subject', {})['repo'] = checkout
    raw['baseline'] = {'kind': 'none'}
    raw.pop('store', None)
    raw['name'] = f'{raw.get("name", "falcon-ci")}-bisect-{sha[:10]}'
    p = os.path.join(os.path.dirname(cfg_path), f'.falcon-bisect-{sha[:12]}.json')
    with open(p, 'w', encoding='utf-8') as fh:
        json.dump(raw, fh, indent=1)
    return p


# ============================================================================================== the whole run
def bisect_stack(repo: str, config: str, good_ref: str, bad_ref: str, out: str, repeats: int = 0,
                 alpha: float = ALPHA, scenarios=(), build: str = '', max_steps: int | None = None,
                 overrides: dict | None = None, keep_checkouts: bool = False, log=None, progress=None,
                 evaluator=None) -> dict:
    """The endpoints, the regressions, the bisection of each (module doc); writes OUT/bisect.json and summary.md and
    returns the document. evaluator: a CIEvaluator (tests pass their own)."""
    from . import provenance as PV
    from .paths import atomic_write
    log = log or (lambda t: None)
    repo = os.path.realpath(repo)
    if not os.path.isdir(os.path.join(repo, '.git')) and not git(repo, 'rev-parse', '--git-dir', check=False):
        raise BisectError(f'{repo} is not a git repository')
    if os.path.isabs(config):
        config = os.path.relpath(config, repo)
    if config.startswith('..'):
        raise BisectError('the CI config must be inside the repository (it is read from each commit)')
    if not 0 < alpha < 0.5:
        raise BisectError('alpha between 0 and 0.5')
    os.makedirs(out, exist_ok=True)
    t0 = time.monotonic()
    good_sha, bad_sha = resolve_ref(repo, good_ref), resolve_ref(repo, bad_ref)
    commits = commits_between(repo, good_sha, bad_sha)
    good_info = commit_info(repo, good_sha)
    ev = evaluator or CIEvaluator(repo, config, out, 1, overrides, build, keep_checkouts, log)
    doc = {'schema': BISECT_SCHEMA, 'repo': repo, 'config': config, 'good': dict(good_info, ref=good_ref),
           'bad': dict(commits[-1], ref=bad_ref), 'commits': commits, 'alpha': alpha, 'build': build or None,
           'started': PV.utc_now(), 'steps': [], 'notes': [], 'skipped': []}

    def note(t):
        doc['notes'].append(t)
        log(t)
    # the endpoints: one run first, to learn whether the mode is deterministic
    ev.repeats = 1
    log(f'good end {good_sha[:10]}')
    g = ev.records(good_sha)
    if g['status'] != 'ok':
        raise BisectError(f'the good end {good_sha[:10]} could not be evaluated: {g.get("why")}')
    det = g.get('deterministic')
    det = True if det is None else bool(det)
    want = int(repeats) if repeats else (1 if det else 5)
    if det and want > 1:
        note(f'the mode is deterministic: a run repeats bit for bit, so --repeats {want} would repeat each outcome; '
             'one run per commit')
        want = 1
    want = min(want, MAX_REPEATS)
    ev.repeats = want
    if want > 1:
        g = ev.records(good_sha)
    if progress:
        progress(0.1, 'good end evaluated')
    log(f'bad end {bad_sha[:10]}')
    b = ev.records(bad_sha)
    if b['status'] != 'ok':
        raise BisectError(f'the bad end {bad_sha[:10]} could not be evaluated: {b.get("why")}')
    if progress:
        progress(0.2, 'bad end evaluated')
    doc.update(deterministic=det, repeats=want,
               rule=('deterministic: a scenario regressed when a run that passed at the good end fails at the bad end; '
                     'a commit is bad when one of those runs fails there' if det else
                     f'Fisher exact test, one-sided, alpha {alpha:g}, over {want} repeat(s) of every run; a commit is '
                     'bad when its pass rate is below the midpoint of the two ends\''))
    tg, tb = tally(g['records']), tally(b['records'])
    rows = find_regressions(tg, tb, det, alpha)
    doc['endpoints'] = rows
    regs = [r for r in rows if r['regressed'] and (not scenarios or r['scenario'] in scenarios)]
    if scenarios:
        unknown = [s for s in scenarios if s not in {r['scenario'] for r in rows}]
        if unknown:
            note(f'not in the suite: {", ".join(unknown)}')
    doc['regressed'] = [r['scenario'] for r in regs]
    doc['bisect'] = {}
    if not regs:
        note('no scenario regressed between the two refs by the rule above: nothing to bisect')
    elif len(commits) == 1:
        for r in regs:
            doc['bisect'][r['scenario']] = {'status': 'found', 'first_bad': commits[0], 'candidates': [commits[0]],
                                            'tested': 0}
    else:
        reg_by = {r['scenario']: r for r in regs}
        n = len(commits)

        def evaluate(i):
            c = commits[i]
            log(f'testing {c["short"]} ({i + 1} of {n}): {c["subject"][:60]}')
            res = ev.records(c['sha'])
            if res['status'] != 'ok':
                doc['skipped'].append({'sha': c['sha'], 'why': res.get('why')})
                return {s: SKIP for s in reg_by}
            t = tally(res['records'])
            return {s: classify(t.get(s), reg, det) for s, reg in reg_by.items()}

        def on_step(st):
            c = commits[st['index']]
            res = ev.records(c['sha'])
            t = tally(res.get('records') or [])
            doc['steps'].append({'sha': c['sha'], 'short': c['short'], 'subject': c['subject'],
                                 'verdicts': st['verdicts'], 'why': res.get('why') or None,
                                 'counts': {s: {'k': t[s]['k'], 'n': t[s]['n']} for s in reg_by if s in t}})
            if progress:
                progress(min(0.2 + 0.8 * len(doc['steps']) / max(1.0, math.log2(n) * len(reg_by) + 1), 0.99),
                         f'{len(doc["steps"])} commits tested')
        out_b = run_bisect(n, list(reg_by), evaluate, max_steps, on_step)
        for t in out_b['notes']:
            note(t)
        for s, r in out_b['result'].items():
            row = dict(r)
            row['first_bad'] = commits[r['first_bad']] if r['first_bad'] is not None else None
            row['last_good'] = commits[r['last_good']] if r['last_good'] >= 0 else good_info
            row['candidates'] = [commits[i] for i in r['candidates']]
            doc['bisect'][s] = row
    doc['finished'] = PV.utc_now()
    doc['wall_s'] = round(time.monotonic() - t0, 2)
    doc['provenance'] = PV.block({}, None, {'repo': repo, 'config': config, 'good': good_sha, 'bad': bad_sha,
                                            'repeats': want, 'alpha': alpha, 'build': build or None})
    PV.write_json(os.path.join(out, 'bisect.json'), doc)
    atomic_write(os.path.join(out, 'summary.md'), markdown(doc))
    if progress:
        progress(1.0, 'done')
    return doc


def markdown(doc: dict) -> str:
    L = [f'# Regression bisect: {doc["good"]["short"]} (good) … {doc["bad"]["short"]} (bad)', '',
         f'{len(doc["commits"])} commits on the first-parent chain; {doc.get("rule", "")}.', '',
         '| scenario | good end | bad end | test | regressed |', '|---|---|---|---|---|']
    for r in doc.get('endpoints') or ():
        f = (lambda e: f'{e["k"]}/{e["n"]}' if e else '—')
        L.append(f'| {r["scenario"]} | {f(r["good"])} | {f(r["bad"])} | {r.get("why", "")} | '
                 f'{"**yes**" if r["regressed"] else "no"} |')
    L += ['']
    for s, r in (doc.get('bisect') or {}).items():
        if r['status'] == 'found' and r['first_bad']:
            c = r['first_bad']
            L.append(f'- **{s}**: first bad commit `{c["short"]}` — {c["subject"]} ({c["author"]}, {c["date"]}); '
                     f'{r.get("tested", 0)} commits tested')
        else:
            L.append(f'- **{s}**: {r["status"]} — one of ' + ', '.join(f'`{c["short"]}`' for c in r['candidates']) +
                     (f' ({r["why"]})' if r.get('why') else ''))
    for sk in doc.get('skipped') or ():
        L.append(f'- skipped `{sk["sha"][:10]}`: {sk["why"]}')
    for n in doc.get('notes') or ():
        L.append(f'- note: {n}')
    return '\n'.join(L) + '\n'


# ============================================================================================== the CLI
def main_bisect(argv) -> int:
    """`falcon bisect run REPO --config F --good A --bad B …` | `falcon bisect show OUT` (docs/bisect.md). 0 done (a
    first bad commit found, ambiguous, or no regression), 2 failed, 3 cancelled."""
    import argparse
    from . import protocol as P
    ap = argparse.ArgumentParser(prog='falcon bisect', description='find the commit of a driving stack that broke a '
                                                                   'scenario (docs/bisect.md)')
    sub = ap.add_subparsers(dest='what', required=True)
    r = sub.add_parser('run', help='run the suite at both refs, then bisect each regressed scenario')
    r.add_argument('repo', help='the stack\'s git repository')
    r.add_argument('--config', required=True, help='its CI config (falcon.ci/1), relative to the repository')
    r.add_argument('--good', required=True, help='a ref where the stack was good (older)')
    r.add_argument('--bad', required=True, help='a ref where it is bad (newer)')
    r.add_argument('--out', default='', help='the output folder (default: falcon-bisect-<good>-<bad> here)')
    r.add_argument('--repeats', type=int, default=0, help='runs of the suite per commit when the mode is not '
                                                          'deterministic (default 5; deterministic: 1)')
    r.add_argument('--alpha', type=float, default=ALPHA, help='the endpoint test\'s level (default 0.05)')
    r.add_argument('--scenario', action='append', default=[], help='bisect only these scenarios')
    r.add_argument('--build', default='', help='a shell command run in each checkout first; failure (or exit 125) '
                                               'skips the commit')
    r.add_argument('--max-steps', type=int, default=0, help='stop after this many commits tested')
    r.add_argument('--fake-carla', default='', help='CI mode against the fake carla (tests)')
    r.add_argument('--keep-checkouts', action='store_true')
    from ..sim.service import placement_arg
    placement_arg(r)
    s = sub.add_parser('show', help='print a bisect\'s summary')
    s.add_argument('out')
    s.add_argument('--json', action='store_true')
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return 0 if e.code == 0 else 2
    if a.what == 'show':
        p = a.out if a.out.endswith('.json') else os.path.join(a.out, 'bisect.json')
        try:
            with open(p, encoding='utf-8') as fh:
                doc = json.load(fh)
        except (OSError, ValueError) as e:
            print(f'failed: {e}', file=sys.stderr)
            return 2
        print(json.dumps(doc, indent=2) if a.json else markdown(doc))
        return 0
    out = os.path.abspath(a.out or f'falcon-bisect-{a.good}-{a.bad}'.replace('/', '_'))
    em = P.Emitter()
    em.hello('bisect')
    try:
        doc = bisect_stack(a.repo, a.config, a.good, a.bad, out, a.repeats, a.alpha, a.scenario, a.build,
                           a.max_steps or None, dict({'fake_carla': a.fake_carla} if a.fake_carla else {},
                                                     **({'placement': a.on} if a.on != 'local' else {})),
                           a.keep_checkouts, log=lambda t: em.log(t),
                           progress=lambda f, t: em.progress(f, t, force=True))
    except BisectError as e:
        print(f'failed: {e}', file=sys.stderr, flush=True)
        return 2
    except KeyboardInterrupt:
        print('cancelled', file=sys.stderr, flush=True)
        return 3
    print(markdown(doc), flush=True)
    em.result('ok', path=out, regressed=doc['regressed'],
              found={s: (r['first_bad'] or {}).get('short') for s, r in doc['bisect'].items()})
    return 0
