"""The planner benchmark's runner (core.planbench): drive each planner over a fixed route set through the job queue,
then score (shared by the GUI and the CLI).

A benchmark is a folder <workspace>/benchmarks/<name>/ holding benchmark.json (the route set and the planners),
runs/<planner>/<route id>/ (one run folder per route: a scenario drive, a stack run) or runs/<planner>/episodes.jsonl
(a policy's episodes in the environment, one per route), index.jsonl (one line per finished route), and, once scored,
scores.json and results.json. Re-running resumes: a route already in index.jsonl as finished is not driven again.

Planners:
    follower, none, autopilot   the scenario director's ego drivers in a scenario drive: Falcon's lane follower (the
                                reference: it sees the scenario's actors, a privileged planner as NAVSIM's PDM-Closed
                                is), constant speed without braking, CARLA's traffic manager (physics: not
                                bit-for-bit repeatable)
    policy:DIR                  a trained policy (core.policy, core.imitation, core.rl) in the environment
                                (core.gymenv), one episode per route with the route's seed
    stack:SUITE.json            the stack of a stack suite (core.stack: its driver, bridge and launch commands; a
                                ROS 2 stack through the bridge, core.stackbridge), one stack run per route

Drive planners record with the rig reduced to one small camera without labels (the planner reads the director's
state, not the sensors) and the map's lanes (lanes.json) for drivable-area and direction checks; a stack keeps the
whole rig, since it perceives.
"""
from __future__ import annotations

import json
import os
import time

from ..core import planbench as PB
from ..core import protocol as P
from ..core import scenarios as SC
from ..core.jobs import FINAL
from ..core.provenance import read_json, utc_now, write_json


def _scenario(project, key):
    scn = project.item('scenarios', key) or (SC.builtin(key) if key in SC.LIBRARY else None)
    if scn is None:
        raise PB.BenchError(f'no scenario {key!r}')
    return scn


def _small_rig(cfg):
    cams = [s for s in cfg['rig'].get('sensors') or [] if s.get('type') in ('camera.pinhole',)]
    if cams:
        c = dict(cams[0], width=64, height=36, labels=[])
        c.pop('faults', None)
        cfg['rig'] = dict(cfg['rig'], sensors=[c])
    cfg['write_lanes'] = True
    return cfg


def drive_config(project, svc, route, world, rig, out, driver):
    from ..core.drive import scenario_config
    cfg = scenario_config(project, _scenario(project, route['scenario']), world, rig, out, svc.port, svc.tm_port,
                          seed=int(route['seed']), render_budget_mpx=float(svc.settings.get('render_budget_mpx')),
                          workspace=project.workspace)
    cfg['scenario']['ego'] = dict(cfg['scenario'].get('ego') or {}, driver=driver)
    cfg['scenario']['motion'] = route.get('motion', 'kinematic')
    cfg['name'] = f'bench {driver} {route["id"]}'
    cfg['bench'] = {'route': route['id'], 'planner': driver}
    return _small_rig(cfg)


class Bench:
    def __init__(self, svc, project, name: str, route_set: str = 'falcon-pdm-v1', route_ids=None, rig: str = '',
                 progress=None, log=None):
        self.svc, self.project, self.name = svc, project, name
        self.dir = PB.bench_dir(project.workspace, name)
        os.makedirs(self.dir, exist_ok=True)
        rs = PB.routes(route_set)
        if route_ids:
            unknown = sorted(set(route_ids) - {r['id'] for r in rs})
            if unknown:
                raise PB.BenchError(f'not in {route_set}: {", ".join(unknown)}')
            rs = [r for r in rs if r['id'] in route_ids]
        self.routes, self.route_set = rs, route_set
        self.rig = rig or project.doc['rigs'][0]['id']
        self.progress, self.log = progress or (lambda f, t: None), log or (lambda t: None)

    def write_doc(self, planners):
        p = os.path.join(self.dir, 'benchmark.json')
        doc = read_json(p) if os.path.isfile(p) else {'schema': PB.SCHEMA, 'name': self.name, 'created': utc_now()}
        doc.update(route_set=self.route_set, about=PB.ROUTE_SET_ABOUT.get(self.route_set), rig=self.rig,
                   reference=PB.REFERENCE, planners=sorted(set(doc.get('planners', [])) | set(planners)),
                   routes=[r['id'] for r in PB.routes(self.route_set)])
        write_json(p, doc)

    def _append(self, row):
        with open(os.path.join(self.dir, 'index.jsonl'), 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(row) + '\n')

    def run(self, planners: list) -> dict:
        """Drive every planner over the routes (finished ones skipped); returns {'done', 'skipped', 'failed'}."""
        self.write_doc(planners)
        have = {(r['planner'], r['route']) for r in PB.read_index(self.dir) if r.get('status', 'ok') == 'ok'}
        out = {'done': 0, 'skipped': [], 'failed': []}
        total = max(len(planners) * len(self.routes), 1)
        k = 0
        for planner in planners:
            todo = []
            for r in self.routes:
                world, why = PB.resolve_world(self.project, r['world'])
                if world is None:
                    out['skipped'].append({'planner': planner, 'route': r['id'], 'why': why})
                elif (planner, r['id']) not in have:
                    todo.append((r, world))
            kind = planner.partition(':')[0]
            if kind == 'policy':
                self.run_policy(planner, todo, out)
            else:
                for r, world in todo:
                    k += 1
                    self.progress(k / total, f'{planner} on {r["id"]}')
                    if kind == 'stack':
                        row = self.run_stack(planner, r, world)
                    elif planner in PB.PLANNERS_DRIVE:
                        row = self.run_drive(planner, r, world)
                    else:
                        raise PB.BenchError(f'unknown planner {planner!r} (follower, none, autopilot, policy:DIR, '
                                            f'stack:SUITE.json)')
                    self._append(row)
                    if row['status'] == 'ok':
                        out['done'] += 1
                    else:
                        out['failed'].append(row)
        return out

    def _out(self, planner, route):
        return os.path.join(self.dir, 'runs', PB.planner_slug(planner), route['id'])

    def _wait(self, job):
        while job.state not in FINAL:
            self.svc.queue.poll()
            time.sleep(0.1)

    def run_drive(self, planner, route, world) -> dict:
        out = self._out(planner, route)
        cfg = drive_config(self.project, self.svc, route, world, self.rig, out, planner)
        job = self.svc.submit_drive(cfg, title=cfg['name'])
        self._wait(job)
        return self._row(planner, route, out)

    def _row(self, planner, route, out):
        try:
            run = read_json(os.path.join(out, 'run.json'))
        except (OSError, ValueError):
            run = {'status': 'failed', 'error': 'no run.json'}
        return {'planner': planner, 'route': route['id'], 'run': os.path.relpath(out, self.dir),
                'status': 'ok' if run.get('status') == 'ok' else 'failed', 'error': run.get('error')}

    def run_stack(self, planner, route, world) -> dict:
        from ..core import stack as ST
        from .stacksuite import run_one
        suite = read_json(planner.partition(':')[2])
        st = suite.get('stack') or {}
        out = self._out(planner, route)
        cfg = ST.stack_config(self.project, world, suite.get('rig') or self.rig, out, self.svc.port,
                              self.svc.tm_port, driver=st.get('driver', 'ros2'),
                              scenario=_scenario(self.project, route['scenario']), seed=int(route['seed']),
                              motion=route.get('motion', 'kinematic'), bridge=st.get('bridge'),
                              render_budget_mpx=float(self.svc.settings.get('render_budget_mpx')),
                              workspace=self.project.workspace, name=f'bench {planner} {route["id"]}')
        cfg['write_lanes'] = True
        cfg['bench'] = {'route': route['id'], 'planner': planner}
        run_one(self.svc, cfg, st.get('launch') or None, int(st.get('domain_id', 0)), cfg['name'])
        return self._row(planner, route, out)

    def run_policy(self, planner, todo, out):
        """One environment per scenario world, an episode per route with the route's seed (core.leaderboard's way)."""
        if not todo:
            return
        from ..core.gymenv import FalconDriveEnv, gym_config
        from ..core.policy import load_policy
        pol = load_policy(planner.partition(':')[2])
        bev = pol.doc.get('bev') if pol.uses_bev else None
        cam = pol.doc.get('camera') if getattr(pol, 'uses_camera', False) else None
        pdir = os.path.join(self.dir, 'runs', PB.planner_slug(planner))
        os.makedirs(pdir, exist_ok=True)
        eps_path = os.path.join(pdir, 'episodes.jsonl')
        groups = {}
        for r, world in todo:
            groups.setdefault((world, r['scenario']), []).append(r)
        for (world, key), rs in groups.items():
            env_dir = os.path.join(pdir, 'env', f'{world}-{key}')
            env = None
            try:
                cfg = gym_config(self.project, world, env_dir, self.svc.port, self.svc.tm_port, rig_id=self.rig,
                                 seed=int(rs[0]['seed']), scenario=_scenario(self.project, key),
                                 action=pol.action['kind'], bev=bev, camera=cam, log_actions=True,
                                 workspace=self.project.workspace, name=f'bench {planner} {key}')
                cfg['write_lanes'] = True
                env = FalconDriveEnv(cfg, self.svc)
                names = env.feature_names
                for r in rs:
                    obs, info = env.reset(seed=int(r['seed']))
                    done = False
                    while not done:
                        obs, _rew, term, trunc, info = env.step(pol.act(obs, names))
                        done = term or trunc
                    line = self._episode_line(env.run_dir, info.get('episode_summary') or {})
                    n = self._count(eps_path)
                    with open(eps_path, 'a', encoding='utf-8') as fh:
                        fh.write(json.dumps(line, separators=(',', ':')) + '\n')
                    self._append({'planner': planner, 'route': r['id'], 'episodes': os.path.relpath(eps_path, self.dir),
                                  'episode': n, 'env_dir': env.run_dir, 'status': 'ok'})
                    out['done'] += 1
            except Exception as e:          # noqa: BLE001 (a world the env cannot host, a lost server: recorded)
                for r in rs:
                    row = {'planner': planner, 'route': r['id'], 'status': 'failed',
                           'error': f'{type(e).__name__}: {e}'}
                    self._append(row)
                    out['failed'].append(row)
            finally:
                if env is not None:
                    env.close()

    @staticmethod
    def _count(path):
        if not os.path.isfile(path):
            return 0
        with open(path, encoding='utf-8') as fh:
            return sum(1 for x in fh if x.strip())

    @staticmethod
    def _episode_line(run_dir, summ):
        """The worker's full line of the episode (with its trajectories), matched by episode number and seed."""
        p = os.path.join(run_dir, 'episodes.jsonl')
        try:
            with open(p, encoding='utf-8') as fh:
                for line in fh:
                    row = json.loads(line)
                    if row.get('episode') == summ.get('episode') and row.get('seed') == summ.get('seed'):
                        return row
        except (OSError, ValueError):
            pass
        return dict(summ)


def run_argv(project_path: str, name: str, planners: list, route_set: str = 'falcon-pdm-v1', routes=None,
             fake_carla: str = '') -> list:
    """`falcon bench run …` as the Scenarios page queues it (the arguments after `python -m falcon_autonomy`)."""
    argv = ['bench', 'run', project_path, '--name', name, '--set', route_set]
    for p in planners:
        argv += ['--planner', p]
    for r in routes or ():
        argv += ['--route', r]
    return argv + (['--fake-carla', fake_carla] if fake_carla else [])


def main_run(a) -> int:
    """`falcon bench run PROJECT --name N --planner P …` (core.planbench.main_bench parsed it)."""
    from ..core.jobs import JobQueue, SubprocessLauncher
    from ..core.project import Project, ProjectError
    from ..core.settings import Settings
    from .service import CarlaService
    try:
        proj = Project.load(a.project)
    except (OSError, ProjectError) as e:
        print(f'failed: {e}', flush=True)
        return P.FAILED
    planners = list(dict.fromkeys(a.planner))
    if not a.no_reference and PB.REFERENCE not in planners:
        planners.insert(0, PB.REFERENCE)                # EP's reference: the lane follower on every route
    svc = CarlaService(Settings(), JobQueue(SubprocessLauncher()), proj.workspace, fake_carla=a.fake_carla)
    em = P.Emitter()
    em.hello('bench')
    try:
        b = Bench(svc, proj, a.name, a.set, a.route, a.rig, lambda f, t: em.progress(f, t, force=True), em.log)
        out = b.run(planners)
        for s in out['skipped']:
            em.log(f'skipped {s["route"]}: {s["why"]}')
        res = PB.results(b.dir, a.boot)
        print(PB.table(res), flush=True)
        em.result('ok', summary={'driven': out['done'], 'skipped': len(out['skipped']), 'failed': len(out['failed']),
                                 **{p: round(v['metrics']['pdms']['value'], 4) for p, v in res['planners'].items()
                                    if (v['metrics'].get('pdms') or {}).get('value') is not None}})
        return P.OK
    except (PB.BenchError, RuntimeError, ValueError, OSError) as e:
        print(f'failed: {e}', flush=True)
        return P.FAILED
    finally:
        svc.shutdown()
