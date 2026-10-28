"""Stack runs and suites through the job queue (shared by the GUI and the CLI), and the `falcon stack` command.

A stack run is a job running the stack worker (falcon_autonomy.workers.stack). For a ROS 2 stack the run also needs
the bridge node and the stack under test: the suite's "launch" commands (core.stack) are started when the run's job
starts — the worker listens before it connects to CARLA, the node retries until it does — and stopped (SIGTERM; a
docker container named in the command is removed with `docker rm -f`) when the job ends. Their output goes to
<run>/launch-<k>.log.

A suite (SuiteRunner) runs its plan in order, skips the runs results.jsonl already holds as finished, appends a line
per finished run, and writes results.json (the summary with its intervals), results.csv (the table) and run.json
(provenance: the suite's hash, the project, the settings, every run's key). Re-running resumes; a run that did not
finish is run again.

    falcon stack topics PROJECT [--rig R] [--control autoware|ackermann|carla]   the topic plan and Autoware's names
    falcon stack image                          build the bridge image (falcon-ros2-bridge:humble) with docker
    falcon stack bridge --port P [--native] [--domain D]     print the bridge node's command
    falcon stack run PROJECT [--scenario S --seed N] [--driver D] [--test-stack] [--out DIR]   one run
    falcon stack suite new NAME --out SUITE.json [--driver D] [--scenarios a,b] [--seeds 0,1,2] [--test-stack]
    falcon stack suite run PROJECT SUITE.json [--dir DIR]                         a batch (resumable)
    falcon stack replay RUN [--into DIR]        re-run a run with the controls it recorded, compare the trajectories
    falcon stack odd new --out ODD.json         an ODD description to edit (core.safetycase)
    falcon stack report SUITE_DIR --odd ODD.json --out DIR [--study DIR]... [--max-failure-rate F]   the safety case

Exit codes as the CLI's: 0 ok (for a suite: it ran, whatever its pass rate), 2 failed or a usage error, 3 cancelled.
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import time

from ..core import protocol as P
from ..core import stack as ST
from ..core.jobs import FINAL, RUNNING
from ..core.paths import ensure_dir
from ..core.provenance import block, read_json, sha256_json, utc_now, write_json


def submit_stack(svc, cfg: dict, title: str = ''):
    """Write cfg into its run folder (stack.json) and queue the stack worker."""
    run_dir = ensure_dir(cfg['out'])
    path = os.path.join(run_dir, 'stack.json')
    write_json(path, cfg)
    spec = svc.worker_spec('stack', [path], title or cfg.get('name', 'stack run'), 'stack', run_dir,
                           meta={'config': path})
    return svc.queue.submit(spec)


def fill(argv: list, values: dict) -> list:
    out = []
    for a in argv:
        s = str(a)
        for k, v in values.items():
            s = s.replace('{' + k + '}', str(v))
        out.append(s)
    return out


def test_stack_launch(control: str = 'autoware', speed_mps: float = 8.0, udp_only: bool = True) -> list:
    """The launch commands of Falcon's test stack: the bridge node and the follower, each in the bridge image."""
    from ..core import stackbridge as SB
    b = SB.bridge_command('{port}', name='{name}-bridge', domain_id=0, udp_only=udp_only)
    f = SB.follower_command(control, speed_mps, name='{name}-follower', domain_id=0, udp_only=udp_only)
    for argv in (b, f):
        argv[argv.index('ROS_DOMAIN_ID=0')] = 'ROS_DOMAIN_ID={domain}'
    return [b, f]


class Launched:
    """The launch commands of one run, started together and stopped together."""

    def __init__(self, cmds: list, values: dict, log_dir: str):
        self.procs = []
        self.names = []
        for k, argv in enumerate(cmds):
            a = fill(argv, values)
            if 'docker' in os.path.basename(a[0]) and '--name' in a:
                self.names.append(a[a.index('--name') + 1])
            fh = open(os.path.join(log_dir, f'launch-{k}.log'), 'w', encoding='utf-8')
            fh.write('$ ' + ' '.join(a) + '\n')
            fh.flush()
            self.procs.append((subprocess.Popen(a, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                                start_new_session=True), fh))

    def stop(self, grace_s: float = 10.0):
        for p, _ in self.procs:
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        t0 = time.monotonic()
        for p, fh in self.procs:
            try:
                p.wait(max(grace_s - (time.monotonic() - t0), 0.1))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                p.wait(5)
            fh.close()
        for n in self.names:
            subprocess.run(['docker', 'rm', '-f', n], capture_output=True, timeout=30)


def run_one(svc, cfg: dict, launch: list | None = None, domain_id: int = 0, title: str = '', poll=None) -> dict:
    """Run one stack config to the end (launching the ROS 2 side when given); returns its run.json (or a failure).
    A ROS 2 run whose bridge port is 0 gets a free port here."""
    if cfg.get('driver') == 'ros2' and not int((cfg.get('bridge') or {}).get('port') or 0):
        from ..core.stackbridge import free_port
        cfg = dict(cfg, bridge=dict(cfg['bridge'], port=free_port(cfg['bridge'].get('host', '127.0.0.1'))))
    job = submit_stack(svc, cfg, title)
    started = None
    try:
        while job.state not in FINAL:
            svc.queue.poll()
            if launch and started is None and job.state == RUNNING:
                tag = cfg.get('cache_key', '')[:10] or re.sub(r'[^A-Za-z0-9_.-]+', '-',
                                                               os.path.basename(os.path.normpath(cfg['out'])))
                started = Launched(launch, {'port': cfg['bridge']['port'], 'domain': int(domain_id),
                                            'name': f'falcon-{tag}'[:60],
                                            'run': cfg['out']}, cfg['out'])
            if poll:
                poll(job)
            time.sleep(0.1)
    finally:
        if started is not None:
            started.stop()
    try:
        run = read_json(os.path.join(cfg['out'], 'run.json'))
    except (OSError, ValueError):
        run = {'status': 'failed', 'error': f'no run.json ({P.describe_exit(job.exit_code)})'}
    run['_job_state'] = job.state
    return run


class SuiteRunner:
    def __init__(self, svc, project, doc: dict, suite_dir: str, progress=None):
        self.svc, self.project, self.doc, self.dir = svc, project, doc, ensure_dir(suite_dir)
        self.progress = progress or (lambda f, text: None)

    def run(self) -> dict:
        doc = self.doc
        bad = ST.validate_suite(doc, self.project)
        if bad:
            raise ValueError('; '.join(bad))
        write_json(os.path.join(self.dir, 'suite.json'), doc)
        plan = ST.plan_runs(doc, self.project, self.dir, self.svc.port, self.svc.tm_port,
                            float(self.svc.settings.get('render_budget_mpx')))
        done = {r['key'] for r in ST.read_results(self.dir) if r.get('status') == 'ok'}
        st = doc['stack']
        launch = (st.get('launch') or []) if st['driver'] == 'ros2' else []
        t0 = utc_now()
        todo = [(s, c) for s, c in plan if s['key'] not in done]
        for k, (spec, cfg) in enumerate(todo):
            self.progress(k / max(len(todo), 1), f'{spec["scenario"]} seed {spec["seed"]} ({k + 1} of {len(todo)}; '
                                                 f'{len(plan) - len(todo)} done before)')
            run = run_one(self.svc, cfg, launch, int(st.get('domain_id', 0)), cfg['name'])
            if run.get('_job_state') == 'cancelled':
                raise KeyboardInterrupt
            row = ST.result_row(spec, run)
            with open(os.path.join(self.dir, 'results.jsonl'), 'a', encoding='utf-8') as fh:
                fh.write(json.dumps(row) + '\n')
        rows = ST.read_results(self.dir)
        keys = {s['key'] for s, _ in plan}
        rows = [r for r in rows if r['key'] in keys]
        summ = ST.summarise(rows, seed=0)
        summ.update(name=doc['name'], driver=st['driver'], planned=len(plan),
                    finished=sum(1 for r in rows if r.get('status') == 'ok'))
        summ['rules'] = ST.rule_summary(rows, doc.get('rules') or None)
        write_json(os.path.join(self.dir, 'results.json'), summ)
        ST.write_table(rows, os.path.join(self.dir, 'results.csv'))
        write_json(os.path.join(self.dir, 'run.json'), {
            'schema': 'falcon.run/1', 'worker': 'stack-suite', 'status': 'ok', 'exit_code': 0, 'started': t0,
            'finished': utc_now(), 'error': None, 'name': doc['name'], 'suite_sha256': sha256_json(doc),
            'runs': [s['key'] for s, _ in plan],
            'provenance': block({'suite': os.path.join(self.dir, 'suite.json'), 'project': self.project.path},
                                seed=None, settings={k: self.svc.settings.get(k) for k in
                                                     ('rpc_port', 'server_memory_gb', 'server_quality',
                                                      'render_budget_mpx')}),
            'summary': {k: summ['overall'].get(k) for k in ('runs', 'passed', 'failed', 'pass_rate',
                                                             'pass_rate_ci95')}})
        self.progress(1.0, f'{summ["finished"]} of {len(plan)} runs finished')
        return summ


def replay_config(src: str, into: str) -> dict:
    cfg = read_json(os.path.join(src, 'stack.json'))
    cfg = dict(cfg, driver='replay', replay={'run': os.path.abspath(src)}, out=os.path.abspath(into),
               name=f'replay of {os.path.basename(os.path.normpath(src))}')
    return cfg


def compare_trajectories(a_dir: str, b_dir: str) -> dict:
    """The largest difference of any actor's (x, y) between two runs' trajectories.json, metres."""
    import numpy as np
    A = read_json(os.path.join(a_dir, 'trajectories.json'))['actors']
    Bt = read_json(os.path.join(b_dir, 'trajectories.json'))['actors']
    worst = 0.0
    per = {}
    for k in sorted(set(A) & set(Bt)):
        a, b = np.asarray(A[k], float), np.asarray(Bt[k], float)
        n = min(len(a), len(b))
        d = float(np.max(np.hypot(a[:n, 1] - b[:n, 1], a[:n, 2] - b[:n, 2]))) if n else 0.0
        per[k] = {'max_m': d, 'rows': [len(a), len(b)]}
        worst = max(worst, d)
    return {'max_m': worst, 'actors': per, 'same_actors': set(A) == set(Bt)}


# ------------------------------------------------------------------------------------------------------------- CLI
def main_stack(argv) -> int:
    import argparse
    from ..core import stackbridge as SB
    ap = argparse.ArgumentParser(prog='falcon stack', description='stack testing: ROS 2 bridge, suites, safety case')
    ap.add_argument('--fake-carla', default='')
    sub = ap.add_subparsers(dest='what', required=True)
    t = sub.add_parser('topics')
    t.add_argument('project')
    t.add_argument('--rig', default='')
    t.add_argument('--control', default='autoware', choices=sorted(SB.CONTROL_TYPES))
    sub.add_parser('image')
    b = sub.add_parser('bridge')
    b.add_argument('--port', type=int, required=True, help='the port given to `falcon stack run --port`')
    b.add_argument('--native', action='store_true')
    b.add_argument('--domain', type=int, default=0)
    r = sub.add_parser('run')
    r.add_argument('project')
    r.add_argument('--scenario', default='')
    r.add_argument('--seed', type=int, default=0)
    r.add_argument('--driver', default='ros2', choices=ST.DRIVERS)
    r.add_argument('--world', default='')
    r.add_argument('--rig', default='')
    r.add_argument('--seconds', type=float, default=20.0)
    r.add_argument('--fps', type=float, default=20.0)
    r.add_argument('--motion', default='kinematic', choices=('kinematic', 'physics'))
    r.add_argument('--control', default='autoware', choices=sorted(SB.CONTROL_TYPES))
    r.add_argument('--port', type=int, default=0, help='the bridge link\'s port (0: a free one; give one to start '
                                                     'your own bridge node)')
    r.add_argument('--deadline', type=float, default=0.5)
    r.add_argument('--domain', type=int, default=0)
    r.add_argument('--test-stack', action='store_true', help='launch the bridge node and the test follower')
    r.add_argument('--speed', type=float, default=8.0, help='the test follower\'s cruise speed, m/s')
    r.add_argument('--dds-shm', action='store_true', help='the test stack on Fast DDS\'s default transports (shared '
                                                          'memory and UDP) instead of UDP only')
    r.add_argument('--out', default='')
    s = sub.add_parser('suite')
    s.add_argument('action', choices=('new', 'run'))
    s.add_argument('args', nargs='+', help='new: NAME; run: PROJECT SUITE.json')
    s.add_argument('--out', default='')
    s.add_argument('--dir', default='')
    s.add_argument('--driver', default='follower', choices=ST.DRIVERS)
    s.add_argument('--scenarios', default='lead_brake,stalled_vehicle')
    s.add_argument('--seeds', default='0,1,2')
    s.add_argument('--test-stack', action='store_true')
    s.add_argument('--control', default='autoware', choices=sorted(SB.CONTROL_TYPES))
    s.add_argument('--speed', type=float, default=8.0)
    rp = sub.add_parser('replay')
    rp.add_argument('run')
    rp.add_argument('--into', default='')
    o = sub.add_parser('odd')
    o.add_argument('action', choices=('new',))
    o.add_argument('--out', required=True)
    rep = sub.add_parser('report')
    rep.add_argument('suite_dir')
    rep.add_argument('--odd', required=True)
    rep.add_argument('--out', required=True)
    rep.add_argument('--study', action='append', default=[])
    rep.add_argument('--max-failure-rate', type=float, default=0.05,
                     help='above this (Wilson upper bound) a scenario\'s evidence is called insufficient')
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return 0 if e.code == 0 else P.FAILED
    try:
        return _dispatch(a)
    except (ValueError, OSError, RuntimeError) as e:
        print(f'failed: {e}', flush=True)
        return P.FAILED
    except KeyboardInterrupt:
        print('cancelled', flush=True)
        return P.CANCELLED


def _project(path):
    from ..core.project import Project
    return Project.load(path)


def _service(project_path, fake):
    from ..core.jobs import JobQueue, SubprocessLauncher
    from ..core.paths import workspace_for
    from ..core.settings import Settings
    from .service import CarlaService
    return CarlaService(Settings(), JobQueue(SubprocessLauncher()), workspace_for(project_path) if project_path
                        else '', fake_carla=fake)


def _dispatch(a) -> int:
    from ..core import stackbridge as SB
    em = P.Emitter()
    if a.what == 'topics':
        proj = _project(a.project)
        rig = proj.item('rigs', a.rig) if a.rig else proj.doc['rigs'][0]
        bridge = dict(SB.bridge_defaults(), control=a.control)
        for e in SB.topic_plan(rig, bridge):
            print(f'  publish  {e["topic"]:48s} {e["type"]}')
        c = SB.control_spec(bridge)
        print(f'  control  {c["topic"]:48s} {c["type"]}')
        print('\nAutoware (not run here; check against your Autoware version):')
        for f, ft, aw, at, note in SB.AUTOWARE_MAPPING:
            print(f'  {f:40s} -> {aw:44s} {note}')
        print('\nremaps for this rig (bridge "topics"):', json.dumps(SB.autoware_remaps(rig), indent=2))
        return 0
    if a.what == 'image':
        cmd = ['docker', 'build', '-t', SB.IMAGE, SB.ros_source()]
        print('$ ' + SB.shell(cmd), flush=True)
        return subprocess.call(cmd)
    if a.what == 'bridge':
        print(SB.shell(SB.bridge_command(a.port, domain_id=a.domain, docker=not a.native)))
        return 0
    if a.what == 'odd':
        from ..core.safetycase import example_odd
        write_json(a.out, example_odd())
        print(a.out)
        return 0
    if a.what == 'report':
        from ..core.safetycase import build_report
        rep = build_report(a.suite_dir, read_json(a.odd), a.out, studies=a.study, odd_path=a.odd,
                           max_failure_rate=a.max_failure_rate)
        print(os.path.join(a.out, 'safety-case.html'))
        print(f'{rep["results"]["overall"]["runs"]} runs, ODD coverage {rep["coverage"]["share"]:.0%} of values, '
              f'{len(rep["residual_risks"])} residual risks listed')
        return 0
    if a.what == 'suite' and a.action == 'new':
        if len(a.args) != 1 or not a.out:
            raise ValueError('falcon stack suite new NAME --out SUITE.json')
        doc = ST.new_suite(a.args[0], a.driver, [x for x in a.scenarios.split(',') if x],
                           [int(x) for x in a.seeds.split(',') if x])
        if a.driver == 'ros2':
            doc['stack']['bridge'] = {'control': a.control}
            if a.test_stack:
                doc['stack']['launch'] = test_stack_launch(a.control, a.speed)
        write_json(a.out, doc)
        print(a.out)
        return 0
    if a.what == 'replay':
        src = os.path.abspath(a.run)
        into = os.path.abspath(a.into or src.rstrip('/') + '-replay')
        if os.path.exists(into):
            raise ValueError(f'{into} exists')
        cfg = replay_config(src, into)
        svc = _service(None, a.fake_carla)
        try:
            run = run_one(svc, cfg, title=cfg['name'])
        finally:
            svc.shutdown()
        if run.get('status') != 'ok':
            print(f'failed: {run.get("error")}', flush=True)
            return P.FAILED
        if os.path.isfile(os.path.join(src, 'trajectories.json')):
            cmp = compare_trajectories(src, into)
            write_json(os.path.join(into, 'replay.json'), dict(cmp, original=src, replay=into))
            print(f'largest difference of any actor: {cmp["max_m"]:.6g} m')
        print(into)
        return 0
    if a.what == 'run':
        proj = _project(a.project)
        svc = _service(proj.path, a.fake_carla)
        try:
            from ..core.drive import run_dir_name
            scn = ST.resolve_scenario(a.scenario, proj) if a.scenario else None
            if a.scenario and scn is None:
                raise ValueError(f'no scenario {a.scenario!r}')
            out = a.out or os.path.join(svc.runs_dir(), run_dir_name(f'stack-{a.scenario or "free"}'))
            cfg = ST.stack_config(proj, a.world or proj.doc['worlds'][0]['id'], a.rig or proj.doc['rigs'][0]['id'],
                                  out, svc.port, svc.tm_port, a.driver, scn, a.seed, seconds=a.seconds, fps=a.fps,
                                  motion=a.motion, bridge={'control': a.control, 'port': a.port,
                                                           'deadline_s': a.deadline},
                                  render_budget_mpx=float(svc.settings.get('render_budget_mpx')),
                                  workspace=proj.workspace)
            launch = []
            if a.driver == 'ros2' and a.test_stack:
                launch = test_stack_launch(a.control, a.speed, udp_only=not a.dds_shm)
            elif a.driver == 'ros2':
                if not a.port:
                    raise ValueError('give --port P to run with your own bridge node and stack (or --test-stack)')
                print('start the bridge node and your stack now:\n  ' +
                      SB.shell(SB.bridge_command(a.port, domain_id=a.domain)), flush=True)
            run = run_one(svc, cfg, launch, a.domain, poll=lambda j: em.progress(j.progress, j.text))
        finally:
            svc.shutdown()
        print(json.dumps({k: run.get(k) for k in ('status', 'error', 'summary')}, indent=2))
        print(out)
        return 0 if run.get('status') == 'ok' else P.FAILED
    if a.what == 'suite' and a.action == 'run':
        if len(a.args) != 2:
            raise ValueError('falcon stack suite run PROJECT SUITE.json [--dir DIR]')
        proj = _project(a.args[0])
        doc = read_json(a.args[1])
        d = a.dir or os.path.join(proj.workspace, 'suites', doc.get('name', 'suite'))
        svc = _service(proj.path, a.fake_carla)
        em.hello('stack-suite')
        try:
            summ = SuiteRunner(svc, proj, doc, d, lambda f, text: em.progress(f, text, force=True)).run()
        finally:
            svc.shutdown()
        o = summ['overall']
        print(f'{d}\n{o["runs"]} runs finished, {o["passed"]} passed'
              + (f' ({o["pass_rate"]:.0%}, 95 % interval {o["pass_rate_ci95"][0]:.0%}–{o["pass_rate_ci95"][1]:.0%})'
                 if o['runs'] else ''))
        em.result('ok', summary={'dir': d, 'runs': o['runs'], 'passed': o['passed']})
        return 0
    raise ValueError(f'unknown command {a.what}')
