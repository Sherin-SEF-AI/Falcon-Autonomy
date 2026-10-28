"""Scenario sweeps and rare-case searches, run unattended through the job queue (shared by the GUI and the CLI).

A study is a folder <workspace>/studies/<name>/ holding study.json (what it is), one scenario drive per evaluated
parameter set in runs/<cache key[:16]>/ (core.datasets.drive_key of the drive's config, so the same parameters on the
same scenario, rig and world are never run twice), and results.jsonl — one line per finished evaluation: the
parameters, the run folder, the criticality summary and pass/fail. Re-running a study resumes it: finished
evaluations are read back, not repeated; a cancelled one is run again.

Sweeps evaluate a fixed set (a grid over chosen parameters, or seeded random samples). Searches (core.search) ask for
parameters in batches, are told the objective of each, and steer towards the critical region; their state is rebuilt
from results.jsonl on resume.
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
    """What a study maximises: the run's criticality scalar (core.criticality), 1 for a collision."""
    c = (result.get('scenario') or {}).get('criticality') or {}
    return float(c.get('criticality', 0.0))


class Study:
    def __init__(self, service, project, scenario: dict, name: str, world: str, rig: str, seed: int = 0,
                 traffic: int = 0):
        self.svc, self.project, self.scn = service, project, scenario
        self.name, self.world, self.rig, self.seed, self.traffic = name, world, rig, int(seed), int(traffic)
        self.dir = study_dir(project.workspace, name)
        os.makedirs(os.path.join(self.dir, 'runs'), exist_ok=True)
        write_json(os.path.join(self.dir, 'study.json'),
                   {'name': name, 'scenario': scenario, 'world': world, 'rig': rig, 'seed': seed, 'traffic': traffic})

    def results(self) -> list:
        p = os.path.join(self.dir, 'results.jsonl')
        if not os.path.isfile(p):
            return []
        with open(p, encoding='utf-8') as fh:
            return [json.loads(x) for x in fh if x.strip()]

    def config(self, values: dict, k: int) -> dict:
        cfg = scenario_config(self.project, self.scn, self.world, self.rig, '/pending', self.svc.port,
                              self.svc.tm_port, seed=self.seed, values=values, traffic_vehicles=self.traffic,
                              render_budget_mpx=float(self.svc.settings.get('render_budget_mpx')),
                              workspace=self.project.workspace)
        key = D.drive_key(cfg)
        cfg['out'] = os.path.join(self.dir, 'runs', key[:16])
        cfg['name'] = f'{self.name} #{k}'
        cfg['cache_key'] = key
        return cfg

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
                res['objective'] = objective(res)
                out[i] = res
                if run.get('status') == 'ok':           # only finished evaluations are kept (a resume re-runs the rest)
                    fh.write(json.dumps(res) + '\n')
        if progress:
            progress(len(self.results()))
        return out

    def sweep(self, sets: list, progress=None) -> list:
        return self.evaluate(sets, progress)

    def search(self, method: str, budget: int, batch: int = 4, threshold: float = 0.5, progress=None) -> dict:
        from ..core import search as S
        space = SC.search_space(self.scn)
        cls = {'random': S.RandomSearch, 'cmaes': S.CMAES, 'bayes': S.BayesOpt, 'adaptive': S.AdaptiveSampling}[method]
        algo = cls(space, seed=self.seed)
        past = [r for r in self.results() if r['status'] == 'ok']
        if past:                                        # resume: tell the algorithm what was already evaluated
            algo.tell([r['values'] for r in past], [r['objective'] for r in past])
        while len(algo.history) < budget:
            ask = algo.ask(min(batch, budget - len(algo.history)))
            res = self.evaluate(ask, progress)
            ok = [(a, r) for a, r in zip(ask, res) if r and r['status'] == 'ok']
            if not ok:
                raise RuntimeError('every evaluation of the batch failed: ' + '; '.join(
                    str(r.get('error')) for r in res if r))
            algo.tell([a for a, _ in ok], [r['objective'] for _, r in ok])
        out = {'method': method, 'budget': budget, 'history': algo.history,
               'failures': S.failures(algo.history, threshold), 'threshold': threshold}
        write_json(os.path.join(self.dir, f'search-{method}.json'), out)
        return out


def main_study(argv) -> int:
    """falcon sweep|search PROJECT --scenario KEY --name NAME [--world W] [--rig R] …  (exit codes as the CLI's)."""
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
    ap.add_argument('--grid', type=int, default=0, help='sweep: values per continuous parameter')
    ap.add_argument('--samples', type=int, default=0, help='sweep: seeded random samples')
    ap.add_argument('--method', default='bayes', choices=('random', 'cmaes', 'bayes', 'adaptive'))
    ap.add_argument('--budget', type=int, default=40)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--fake-carla', default='')
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
    svc = CarlaService(Settings(), JobQueue(SubprocessLauncher()), proj.workspace, fake_carla=a.fake_carla)
    em = P.Emitter()
    em.hello(a.kind)
    st = Study(svc, proj, scn, a.name, a.world or proj.doc['worlds'][0]['id'], a.rig or proj.doc['rigs'][0]['id'],
               a.seed)
    total = a.budget if a.kind == 'search' else None
    try:
        if a.kind == 'sweep':
            sets = grid(SC.search_space(scn), a.grid) if a.grid else random_samples(scn, a.samples or 10, a.seed)
            total = len(sets)
            st.sweep(sets, lambda n: em.progress(min(n / total, 1.0), f'{n} of {total}', force=True))
            em.result('ok', summary={'evaluated': len(st.results())})
        else:
            out = st.search(a.method, a.budget, a.batch, a.threshold,
                            lambda n: em.progress(min(n / total, 1.0), f'{n} of {total}', force=True))
            em.result('ok', summary={'evaluated': len(out['history']), 'failures': len(out['failures'])})
    except (RuntimeError, ValueError, ProjectError) as e:
        print(f'failed: {e}', flush=True)
        svc.shutdown()
        return P.FAILED
    svc.shutdown()
    return P.OK
