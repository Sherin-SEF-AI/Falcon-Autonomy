"""Scenario sweeps and rare-case searches, run unattended through the job queue (shared by the GUI and the CLI).

A study is a folder <workspace>/studies/<name>/ holding study.json (what it is), one scenario drive per evaluated
parameter set in runs/<cache key[:16]>/ (core.datasets.drive_key of the drive's config, so the same parameters on the
same scenario, rig and world are never run twice), and results.jsonl — one line per finished evaluation: the
parameters, the run folder, the criticality summary and pass/fail. Re-running a study resumes it: finished
evaluations are read back, not repeated; a cancelled one is run again.

Sweeps evaluate a fixed set (a grid over chosen parameters, or seeded random samples). Searches (core.search) ask for
parameters in batches, are told the objective of each, and steer towards the critical region; their state is rebuilt
from results.jsonl on resume.

The objective (core.milsearch): the run's criticality (the default), a registered model's perception failure on the
run's labelled frames, or both (their minimum). With a model, each batch's drives record first; then the model runs
on their frames in this process (an ONNX or TorchScript model under the CARLA and GPU locks, so a drive and the model
never run at full load together) and every full score is kept in perception/<run>-<objective key>.json. A study whose
objective or model changed re-scores its finished runs from their frames (they are not driven again). Extra search
dimensions ("weather.<field>") override the run's weather. study.json records the objective and the space.
"""
from __future__ import annotations

import itertools
import json
import os
import time

from ..core import datasets as D
from ..core import protocol as P
from ..core import scenarios as SC
from ..core.drive import scenario_config
from ..core.jobs import FINAL
from ..core.provenance import read_json, write_json


METHOD_NAMES = ('bayes', 'bayes-pof', 'cmaes', 'adaptive', 'random')


def method_class(name: str):
    """A study method's searcher (core.search) by its command-line name."""
    from ..core import search as S
    return {'random': S.RandomSearch, 'cmaes': S.CMAES, 'bayes': S.BayesOpt, 'adaptive': S.AdaptiveSampling,
            'bayes-pof': S.BayesFailure}[name]


def study_dir(workspace, name):
    return os.path.join(workspace, 'studies', name)


def grid(space: list, n: int) -> list:
    """n values per continuous parameter (evenly from lo to hi) and every choice: their product."""
    axes = []
    for p in space:
        if 'choices' in p:
            axes.append([(p['name'], c) for c in p['choices']])
        else:
            k = max(int(n), 1)
            axes.append([(p['name'], p['lo'] + (p['hi'] - p['lo']) * (i / (k - 1) if k > 1 else 0.5))
                         for i in range(k)])
    return [dict(c) for c in itertools.product(*axes)]


def random_samples(scn: dict, n: int, seed: int) -> list:
    return [SC.sample_params(scn, seed * 1000003 + i) for i in range(n)]


def objective(result: dict) -> float:
    """The run's criticality scalar (core.criticality), 1 for a collision: what a study maximises by default."""
    c = (result.get('scenario') or {}).get('criticality') or {}
    return float(c.get('criticality', 0.0))


def perception_summary(sc: dict) -> dict:
    """The compact perception record kept in results.jsonl (the full score is in perception/<run>-<key>.json)."""
    keep = ('value', 'metric', 'perception', 'window_miss', 'lost_lead', 'miss_rate', 'worst_actor', 'note',
            'relevant_object_frames', 'synthetic_model')
    out = {k: sc.get(k) for k in keep}
    out['actors'] = [{k: a.get(k) for k in ('actor', 'class', 't_critical', 'frames', 'window_recall', 'lost_lead',
                                            'detection_delay_s', 'failure', 'note')} for a in sc.get('actors') or ()]
    return out


class Study:
    """A scenario study. objective: a core.milsearch spec (default the criticality); extra_space: more search
    dimensions beyond the scenario's parameters (core.milsearch.weather_space: "weather.<field>")."""

    emit = None                    # a protocol Emitter for the GPU-lock wait messages (the CLI sets it)
    cancel = None                  # a callable: True stops a wait for the GPU lock and the model's scoring

    def __init__(self, service, project, scenario: dict, name: str, world: str, rig: str, seed: int = 0,
                 traffic: int = 0, objective: dict | None = None, extra_space: list | None = None,
                 fps: float | None = None, search_seed: int | None = None):
        from ..core import milsearch as MS
        from ..core import search as S
        self.svc, self.project, self.scn = service, project, scenario
        self.name, self.world, self.rig, self.seed, self.traffic = name, world, rig, int(seed), int(traffic)
        self.search_seed = self.seed if search_seed is None else int(search_seed)   # the searcher's generator
        if fps is not None or not hasattr(self, 'fps'):
            self.fps = float(fps) if fps else None                  # None: the drive default (20)
        self.spec = MS.normalise_spec(objective)
        self.extra = S.normalise_space(extra_space or [])
        names = [p['name'] for p in SC.search_space(scenario)]
        clash = [p['name'] for p in self.extra if p['name'] in names]
        if clash:
            raise ValueError(f'{", ".join(clash)} is already a parameter of the scenario')
        if MS.needs_model(self.spec):
            why = MS.evaluable(self.spec['model'], project.workspace) or MS.check_rig(self._rig(),
                                                                                    self.spec.get('camera'))
            if why:
                raise ValueError(f'the {self.spec["objective"]} objective cannot be used: {why}')
        self.okey = MS.spec_key(self.spec, project.workspace)
        self.dir = study_dir(project.workspace, name)
        os.makedirs(os.path.join(self.dir, 'runs'), exist_ok=True)
        self._write_doc()

    def _rig(self) -> dict:
        return self.project.item('rigs', self.rig) or {}

    def objective_doc(self) -> dict:
        from ..core import milsearch as MS
        d = {'kind': self.spec['objective'], 'key': self.okey, 'about': MS.describe(self.spec)}
        if MS.needs_model(self.spec):
            m = self.spec['model']
            d.update(model={k: m.get(k) for k in ('id', 'name', 'runtime', 'entry', 'path', 'params')},
                     model_sha256=MS.model_fingerprint(m, self.project.workspace),
                     settings={k: v for k, v in self.spec.items() if k not in ('objective', 'model')})
        return d

    def _write_doc(self, **extra):
        write_json(os.path.join(self.dir, 'study.json'),
                   dict({'name': self.name, 'scenario': self.scn, 'world': self.world, 'rig': self.rig,
                         'seed': self.seed, 'search_seed': self.search_seed, 'traffic': self.traffic,
                         'objective': self.objective_doc(),
                         'space': self.space()}, **extra))

    def space(self) -> list:
        """The search space: the scenario's parameters and the extra dimensions."""
        return SC.search_space(self.scn) + list(self.extra)

    def results(self) -> list:
        p = os.path.join(self.dir, 'results.jsonl')
        if not os.path.isfile(p):
            return []
        with open(p, encoding='utf-8') as fh:
            return [json.loads(x) for x in fh if x.strip()]

    def base_config(self, values: dict) -> dict:
        """The scenario run's drive config for the scenario's parameter values (subclasses: the loop's cell)."""
        return scenario_config(self.project, self.scn, self.world, self.rig, '/pending', self.svc.port,
                               self.svc.tm_port, seed=self.seed, values=values, traffic_vehicles=self.traffic,
                               render_budget_mpx=float(self.svc.settings.get('render_budget_mpx')),
                               workspace=self.project.workspace, **({'fps': self.fps} if self.fps else {}))

    def config(self, values: dict, k: int) -> dict:
        from ..core import milsearch as MS
        scn_values, weather = MS.split_values(values)
        cfg = MS.apply_weather(self.base_config(scn_values), weather)
        if MS.needs_model(self.spec) and self.spec['every'] > 1:
            cfg['record_every'] = int(self.spec['every'])
        key = D.drive_key(cfg)
        cfg['out'] = os.path.join(self.dir, 'runs', key[:16])
        cfg['name'] = f'{self.name} #{k}'
        cfg['cache_key'] = key
        return cfg

    def _rewrite(self, rows: list):
        from ..core.paths import atomic_write
        atomic_write(os.path.join(self.dir, 'results.jsonl'), ''.join(json.dumps(r) + '\n' for r in rows))

    def _score(self, rows: list) -> None:
        """Give rows (finished runs) this study's objective; the perception objective runs the model on them (the
        CARLA lock is free by now: their drives have finished) and keeps each full score beside the study."""
        from ..core import milsearch as MS
        need = MS.needs_model(self.spec)
        scores = {}
        if need:
            try:
                scores = MS.score_runs([r['run'] for r in rows], self.spec, self.project.workspace, self.emit,
                                       self.cancel)
            except MS.MilError as e:
                raise RuntimeError(f'scoring the runs with the model failed: {e}') from None
            os.makedirs(os.path.join(self.dir, 'perception'), exist_ok=True)
        for r in rows:
            r['criticality'] = objective(r)
            r['objective_kind'] = self.spec['objective']
            r['perception'] = None
            if need:
                sc = scores[r['run']]
                write_json(os.path.join(self.dir, 'perception', f'{os.path.basename(r["run"])}-{self.okey}.json'),
                           sc)
                r['perception'] = perception_summary(sc)
            r['objective'] = MS.objective_value(self.spec, r['criticality'], r['perception'])
            r['objective_key'] = self.okey

    def _ensure_scored(self, rows: list):
        """Score the rows whose objective is not this study's (new runs, or a study whose model changed)."""
        stale = [r for r in rows if r and r.get('status') == 'ok' and r.get('objective_key', 'criticality') !=
                 self.okey]
        if not stale:
            return
        self._score(stale)
        by_key = {r['key']: r for r in stale}
        self._rewrite([by_key.get(r['key'], r) for r in self.results()])

    def current_results(self) -> list:
        """The finished evaluations this study would make now (same config: the cache key matches; every search
        dimension set), scored with its objective."""
        names = [p['name'] for p in self.space()]
        rows = []
        for r in self.results():
            if r.get('status') != 'ok' or any(n not in (r.get('values') or {}) for n in names):
                continue
            try:
                if self.config(r['values'], 0)['cache_key'] != r['key']:
                    continue
            except (KeyError, ValueError, TypeError):
                continue
            rows.append(r)
        self._ensure_scored(rows)
        by_key = {r['key']: r for r in self.results()}
        return [by_key[r['key']] for r in rows]

    def evaluate(self, batch: list, progress=None) -> list:
        """Run the parameter sets (cached ones read back) and return their results, in order."""
        done = {r['key']: r for r in self.results()}
        out, jobs = [None] * len(batch), []
        for i, values in enumerate(batch):
            cfg = self.config(values, len(done) + i)
            if cfg['cache_key'] in done:
                out[i] = done[cfg['cache_key']]
                continue
            jobs.append((i, cfg, self.svc.submit_drive(cfg, title=cfg['name'])))
        while any(j.state not in FINAL for _, _, j in jobs):
            self.svc.queue.poll()
            time.sleep(0.2)
        with open(os.path.join(self.dir, 'results.jsonl'), 'a', encoding='utf-8') as fh:
            for i, cfg, j in jobs:
                try:
                    run = read_json(os.path.join(cfg['out'], 'run.json'))
                except (OSError, ValueError):
                    run = {'status': 'failed', 'error': 'no run.json'}
                res = {'key': cfg['cache_key'], 'values': batch[i], 'run': cfg['out'], 'status': run.get('status'),
                       'error': run.get('error'), 'scenario': run.get('scenario')}
                res['objective'] = res['criticality'] = objective(res)
                if self.okey == 'criticality':
                    res['objective_key'] = 'criticality'
                out[i] = res
                if run.get('status') == 'ok':           # only finished evaluations are kept (a resume re-runs the rest)
                    fh.write(json.dumps(res) + '\n')
        self._ensure_scored([r for r in out if r])
        by_key = {r['key']: r for r in self.results()}
        out = [by_key.get(r['key'], r) if r else r for r in out]
        if progress:
            progress(len(self.results()))
        return out

    def _resume(self, method, cls, space, opts, past):
        """The searcher, resumed: from its saved state (searcher-<method>.json) when that was made with this
        objective, space and threshold — it then continues exactly as the uninterrupted search, and is told any
        evaluation made since (a sweep) — else rebuilt and told every finished evaluation (a random search rebuilt
        so can ask a point again: it is read back from the cache, not driven)."""
        from ..core import provenance as PV
        from ..core import search as S
        p = os.path.join(self.dir, f'searcher-{method}.json')
        algo = None
        try:
            d = read_json(p)
            st = d['state']
            if d.get('objective_key') == self.okey and d.get('threshold') == opts.get('threshold', d.get('threshold')) \
                    and st['space'] == S.normalise_space(space) and st['seed'] == self.search_seed:
                algo = S.from_state(st)
        except (OSError, ValueError, KeyError):
            algo = None
        if algo is not None:
            names = [q['name'] for q in algo.space]

            def key(v):
                return PV.canonical_json({n: v[n] for n in names})
            by_values = {key(r['values']): r for r in past}
            known = {key(h['params']) for h in algo.history}
            if not known <= set(by_values):
                algo = None                           # its history is not this study's results any more
            else:
                new = [r for k, r in by_values.items() if k not in known]
                if new:
                    algo.tell([r['values'] for r in new], [r['objective'] for r in new])
                return algo
        algo = cls(space, seed=self.search_seed, **opts)
        if past:                                        # resume: tell the algorithm what was already evaluated
            algo.tell([r['values'] for r in past], [r['objective'] for r in past])
        return algo

    def sweep(self, sets: list, progress=None) -> list:
        return self.evaluate(sets, progress)

    def search(self, method: str, budget: int, batch: int = 4, threshold: float = 0.5, progress=None) -> dict:
        from ..core import search as S
        space = self.space()
        cls = method_class(method)
        opts = {'threshold': threshold} if cls in (S.AdaptiveSampling, S.BayesFailure) else {}
        past = self.current_results()
        algo = self._resume(method, cls, space, opts, past)
        state_path = os.path.join(self.dir, f'searcher-{method}.json')
        while len(algo.history) < budget:
            ask = algo.ask(min(batch, budget - len(algo.history)))
            res = self.evaluate(ask, progress)
            ok = [(a, r) for a, r in zip(ask, res) if r and r['status'] == 'ok']
            if not ok:
                raise RuntimeError('every evaluation of the batch failed: ' + '; '.join(
                    str(r.get('error')) for r in res if r))
            algo.tell([a for a, _ in ok], [r['objective'] for _, r in ok])
            write_json(state_path, {'objective_key': self.okey, 'threshold': threshold, 'state': algo.state()})
        out = {'method': method, 'budget': budget, 'history': algo.history,
               'failures': S.failures(algo.history, threshold), 'threshold': threshold,
               'objective': self.objective_doc(), 'space': space}
        write_json(os.path.join(self.dir, f'search-{method}.json'), out)
        return out


def objective_args(ap):
    """The objective's command-line options (falcon search and falcon sweep; docs/rare-cases-mil.md)."""
    from ..core import milsearch as MS
    ap.add_argument('--objective', default='criticality', choices=MS.OBJECTIVES,
                    help='criticality (default), perception (the model\'s failure) or both')
    ap.add_argument('--model', default='', help='a registered model id (perception and both)')
    ap.add_argument('--metric', default='perception', choices=MS.METRICS)
    ap.add_argument('--target-class', action='append', default=[], help='a class whose objects in --range also count')
    ap.add_argument('--range', default='', help='LO,HI metres for the target classes (default 0,50)')
    ap.add_argument('--window', type=float, default=MS.DEFAULTS['window_s'], help='seconds before the critical moment')
    ap.add_argument('--lead', type=float, default=MS.DEFAULTS['lead_s'], help='the warning time needed, seconds')
    ap.add_argument('--iou', type=float, default=MS.DEFAULTS['iou'])
    ap.add_argument('--min-score', type=float, default=MS.DEFAULTS['min_score'])
    ap.add_argument('--every', type=int, default=1, help='record (and score) every n-th step')
    ap.add_argument('--camera', default=None)
    ap.add_argument('--weather', action='append', default=[], metavar='FIELD[:LO:HI]',
                    help='also search a CARLA weather field (e.g. fog_density, or fog_density:20:100)')


def objective_from_args(a, proj) -> tuple:
    """(the objective spec, the extra search dimensions) from objective_args' options; ValueError when unusable."""
    from ..core import milsearch as MS
    spec = {'objective': a.objective, 'metric': a.metric, 'target_classes': a.target_class, 'window_s': a.window,
            'lead_s': a.lead, 'iou': a.iou, 'min_score': a.min_score, 'every': a.every, 'camera': a.camera}
    if a.range:
        spec['range_m'] = [float(x) for x in a.range.split(',')]
    if a.objective != 'criticality':
        m = proj.item('models', a.model) if a.model else None
        if m is None:
            raise ValueError(f'the {a.objective} objective needs --model ID (a registered model): '
                             + (f'no model {a.model!r} in the project' if a.model else 'none given'))
        spec['model'] = m
    try:
        return MS.normalise_spec(spec), MS.weather_space(a.weather)
    except MS.MilError as e:
        raise ValueError(str(e)) from None


def sweep_samples(space: list, scn: dict, n: int, seed: int) -> list:
    """Seeded random samples: the scenario's parameters as random_samples draws them, an extra dimension uniformly
    with a generator seeded by (sample seed, its name) — adding a dimension changes no other draw."""
    import random
    names = {p['name'] for p in SC.search_space(scn)}
    out = random_samples(scn, n, seed)
    for i, v in enumerate(out):
        for p in space:
            if p['name'] not in names and 'choices' not in p:
                v[p['name']] = random.Random(f'{seed * 1000003 + i}:{p["name"]}').uniform(p['lo'], p['hi'])
    return out


def main_study(argv) -> int:
    """falcon sweep|search PROJECT --scenario KEY --name NAME [--world W] [--rig R] [--objective …] …  (exit codes as
    the CLI's)."""
    import argparse
    from ..core.jobs import JobQueue, SubprocessLauncher
    from ..core.project import Project, ProjectError
    from ..core.settings import Settings
    from .service import CarlaService
    ap = argparse.ArgumentParser(prog='falcon sweep|search')
    ap.add_argument('kind', choices=('sweep', 'search'))
    ap.add_argument('project')
    ap.add_argument('--scenario', required=True, help='a built-in (core.scenarios.LIBRARY) or a project scenario id')
    ap.add_argument('--name', required=True)
    ap.add_argument('--world', default='')
    ap.add_argument('--rig', default='')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--search-seed', type=int, default=None, help='the searcher\'s seed (default: --seed)')
    ap.add_argument('--grid', type=int, default=0, help='sweep: values per continuous parameter')
    ap.add_argument('--samples', type=int, default=0, help='sweep: seeded random samples')
    ap.add_argument('--method', default='bayes', choices=METHOD_NAMES,
                    help='bayes-pof: Bayesian optimisation by the probability of failure (maps the failure region)')
    ap.add_argument('--budget', type=int, default=40)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--fps', type=float, default=0.0, help='the simulation step rate (default 20)')
    ap.add_argument('--fake-carla', default='')
    objective_args(ap)
    a = ap.parse_args(argv)
    try:
        proj = Project.load(a.project)
    except (OSError, ProjectError) as e:
        print(f'failed: {e}', flush=True)
        return P.FAILED
    scn = proj.item('scenarios', a.scenario) or (SC.builtin(a.scenario) if a.scenario in SC.LIBRARY else None)
    if scn is None:
        print(f'failed: no scenario {a.scenario!r}', flush=True)
        return P.FAILED
    try:
        spec, extra = objective_from_args(a, proj)
    except ValueError as e:
        print(f'failed: {e}', flush=True)
        return P.REFUSED
    svc = CarlaService(Settings(), JobQueue(SubprocessLauncher()), proj.workspace, fake_carla=a.fake_carla)
    em = P.Emitter()
    em.hello(a.kind)
    try:
        st = Study(svc, proj, scn, a.name, a.world or proj.doc['worlds'][0]['id'],
                   a.rig or proj.doc['rigs'][0]['id'], a.seed, objective=spec, extra_space=extra,
                   fps=a.fps or None, search_seed=a.search_seed)
    except ValueError as e:
        print(f'failed: {e}', flush=True)
        svc.shutdown()
        return P.REFUSED
    st.emit = em
    em.log(f'objective: {st.objective_doc()["about"]}')
    total = a.budget if a.kind == 'search' else None
    try:
        if a.kind == 'sweep':
            sets = grid(st.space(), a.grid) if a.grid else sweep_samples(st.space(), scn, a.samples or 10, a.seed)
            total = len(sets)
            st.sweep(sets, lambda n: em.progress(min(n / total, 1.0), f'{n} of {total}', force=True))
            em.result('ok', summary={'evaluated': len(st.results())})
        else:
            out = st.search(a.method, a.budget, a.batch, a.threshold,
                            lambda n: em.progress(min(n / total, 1.0), f'{n} of {total}', force=True))
            em.result('ok', summary={'evaluated': len(out['history']), 'failures': len(out['failures']),
                                     'objective': spec['objective'],
                                     'best': max((h['value'] for h in out['history']), default=None)})
    except (RuntimeError, ValueError, ProjectError) as e:
        print(f'failed: {e}', flush=True)
        svc.shutdown()
        return P.FAILED
    svc.shutdown()
    return P.OK
