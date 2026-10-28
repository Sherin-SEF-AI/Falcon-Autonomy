"""The command line: every heavy action, headless and CI-friendly, through the same code as the GUI.

    falcon [gui] [PROJECT]                     the application
    falcon doctor                              check the machine and CARLA (no server started)
    falcon project new NAME PATH               a new project file
    falcon project validate PATH               check a project against its schema (and migrate in memory)
    falcon project migrate PATH                migrate a project to this build's format version, in place
    falcon probe [--wait S]                    what server answers on Falcon's port
    falcon smoke PROJECT                       the 30 s smoke drive; records the measured throughput
    falcon drive PROJECT [--world W --rig R --seconds S --fps F --seed N]
    falcon tidy                                tidy Falcon's server after a killed worker
    falcon reproduce RUN [--into DIR]          re-run a run from its own config and compare within a tolerance
    falcon runs PROJECT [--rebuild]            the runs in a project's catalogue
    falcon export RUN... --format F --out DIR  COCO, YOLO, KITTI, nuScenes, waymo, bdd_lanes or mcap
    falcon evaluate PROJECT --model M --run RUN... --out DIR   score a model (or --compare A B)
    falcon gap --sim RUN... --real KIND:PATH --out DIR   the sim-to-real gap of camera images
    falcon sweep PROJECT --scenario S --name N [--grid K | --samples K]   a parameter sweep (resumable)
    falcon search PROJECT --scenario S --name N [--method bayes|cmaes|adaptive|random] [--budget B]   rare cases
    falcon schema                              print the project schema's path

Exit codes: 0 ok; otherwise the worker's code (falcon_autonomy.core.protocol: 2 failed, 3 cancelled, 4 server lost,
5 no server, 6 refused, 7 no carla) or 2 for a usage error. docs/cli.md is the reference.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time

from . import APP_NAME, __version__
from .core import protocol as P


def _settings():
    from .core.settings import Settings
    return Settings()


def _service(project_path: str | None, fake: str = ''):
    from .core.jobs import JobQueue, SubprocessLauncher
    from .core.paths import workspace_for
    from .sim.service import CarlaService
    st = _settings()
    q = JobQueue(SubprocessLauncher())
    ws = workspace_for(project_path) if project_path else ''
    return CarlaService(st, q, ws, fake_carla=fake), st


def _run_job(svc, job, quiet=False) -> int:
    """Wait for a job, printing its progress; Ctrl+C cancels it politely (a second Ctrl+C kills it)."""
    from .core.jobs import FINAL
    state = {'hits': 0}

    def on_int(*_):
        state['hits'] += 1
        if state['hits'] == 1:
            print('\ncancelling (Ctrl+C again to kill)…', file=sys.stderr)
            svc.queue.cancel(job)
        elif job.handle is not None:
            job.handle.kill()
    old = signal.signal(signal.SIGINT, on_int)
    last = ''
    shown = 0
    try:
        while job.state not in FINAL:
            svc.queue.poll()
            line = job.summary()
            if not quiet and line != last:
                print(f'[{job.spec.title}] {line}', file=sys.stderr)
                last = line
            for lvl, text in job.log[shown:]:
                if not quiet and lvl in ('warning', 'error'):
                    print(f'  {lvl}: {text}', file=sys.stderr)
            shown = len(job.log)
            time.sleep(0.25)
    finally:
        signal.signal(signal.SIGINT, old)
    for lvl, text in job.log[shown:]:
        if lvl in ('warning', 'error', 'stderr') and text:
            print(f'  {lvl}: {text}', file=sys.stderr)
    code = job.exit_code if job.exit_code is not None else P.FAILED
    if code < 0:
        code = P.CANCELLED if job.cancel_requested else P.CRASHED
    print(f'[{job.spec.title}] {job.state}: {P.describe_exit(job.exit_code)}', file=sys.stderr)
    return code


def cmd_doctor(a) -> int:
    from .core import sysinfo
    from .core.paths import tilde
    from .sim import server as S
    st = _settings()
    ok_all = True

    def row(name, ok, detail):
        nonlocal ok_all
        ok_all &= bool(ok)
        print(f'  {"ok " if ok else "-- "} {name:22s} {detail}')
    print(f'{APP_NAME} {__version__}')
    root = st.get('carla_root') or (S.candidate_roots() or [''])[0]
    row('CARLA folder', S.is_carla_root(root), f'{tilde(root)} ({S.carla_version(root) or "?"})' if root else
        'not found: set it in Settings')
    py = st.get('client_python') or S.client_python(root)
    chk = S.check_client(py) if py else {'ok': False, 'text': 'no venv beside the CARLA folder'}
    row('CARLA\'s Python', chk['ok'], f'{tilde(py)}: {chk["text"]}' if py else chk['text'])
    for name, ok, detail in sysinfo.checks():
        row(name, ok, detail)
    port = int(st.get('rpc_port'))
    used = [p for p in range(port, port + 3) if S.port_open('127.0.0.1', p)]
    row('Falcon\'s ports', not used, f'RPC {port}, traffic manager {st.get("tm_port")}' +
        (f' — in use: {used}' if used else ' (free)'))
    return 0 if ok_all else P.FAILED


def cmd_project(a) -> int:
    from .core.project import Project, ProjectError, migrate, validate
    if a.action == 'new':
        p = Project.new(a.name)
        p.save(a.path)
        print(p.path)
        return 0
    try:
        with open(a.path, encoding='utf-8') as fh:
            doc = json.load(fh)
        doc, notes = migrate(doc)
    except (OSError, ValueError, ProjectError) as e:
        print(f'failed: {e}', file=sys.stderr)
        return P.FAILED
    errs = validate(doc)
    for n in notes:
        print(n)
    if errs:
        print('\n'.join(errs), file=sys.stderr)
        return P.FAILED
    if a.action == 'migrate' and notes:
        p = Project(doc, a.path)
        p.save(a.path)
        print(f'saved {a.path}')
    elif a.action == 'validate':
        print('valid')
    return 0


def _load_project(path):
    from .core.project import Project, ProjectError
    try:
        return Project.load(path)
    except (OSError, ProjectError) as e:
        print(f'failed: {e}', file=sys.stderr)
        sys.exit(P.FAILED)


def cmd_drive(a, smoke=False) -> int:
    from .core.drive import drive_config, run_dir_name, smoke_config
    from .core.project import ProjectError
    proj = _load_project(a.project)
    svc, st = _service(proj.path, a.fake_carla)
    try:
        if smoke:
            out = a.out or os.path.join(svc.runs_dir(), run_dir_name('smoke'))
            cfg = smoke_config(proj, out, svc.port, svc.tm_port, a.seed)
        else:
            world = a.world or proj.doc['worlds'][0]['id']
            rig = a.rig or proj.doc['rigs'][0]['id']
            out = a.out or os.path.join(svc.runs_dir(), run_dir_name(f'drive-{world}'))
            cfg = drive_config(proj, world, rig, out, svc.port, svc.tm_port, a.seconds, a.fps, a.seed,
                               render_budget_mpx=float(st.get('render_budget_mpx')))
        if a.max_wall:
            cfg['max_wall_s'] = a.max_wall
        job = svc.submit_drive(cfg, kind='smoke' if smoke else 'drive')
        code = _run_job(svc, job)
        if code == P.OK:
            print(out)
            if smoke and not a.fake_carla:
                print(f'throughput: {svc.throughput().source}')
        return code
    except (ProjectError, RuntimeError) as e:
        print(f'failed: {e}', file=sys.stderr)
        return P.FAILED
    finally:
        if not a.keep_server:
            svc.shutdown()


def cmd_probe(a) -> int:
    svc, _ = _service(None, a.fake_carla)
    try:
        job = svc.submit_probe(a.wait)
        code = _run_job(svc, job, quiet=True)
        if job.result:
            print(json.dumps(job.result.get('summary') or job.result, indent=2))
        return code
    finally:
        if not a.keep_server:
            svc.shutdown()


def cmd_tidy(a) -> int:
    svc, _ = _service(None, a.fake_carla)
    return _run_job(svc, svc.submit_tidy())


def cmd_reproduce(a) -> int:
    from .core.provenance import read_json, write_json
    from .core.reproduce import compare_drives
    src = os.path.abspath(a.run)
    cfg_path = os.path.join(src, 'drive.json')
    if not os.path.isfile(cfg_path):
        print(f'failed: {src} has no drive.json (only drives can be reproduced in this version)', file=sys.stderr)
        return P.FAILED
    cfg = read_json(cfg_path)
    into = os.path.abspath(a.into or src.rstrip('/') + '-reproduced')
    if os.path.exists(into):
        print(f'failed: {into} exists', file=sys.stderr)
        return P.FAILED
    cfg['out'] = into
    svc, st = _service(None, a.fake_carla)
    svc.workspace = os.path.dirname(src)
    try:
        code = _run_job(svc, svc.submit_drive(cfg, title=f'Reproduce {os.path.basename(src)}'))
    finally:
        svc.shutdown()
    if code != P.OK:
        return code
    rep = compare_drives(src, into, {'pose_tol': a.pose_tol, 'rgb_tol': a.rgb_tol})
    rep['original'], rep['reproduction'] = src, into
    write_json(os.path.join(into, 'reproduce.json'), rep)
    for c in rep['checks']:
        print(f'  {"ok " if c["ok"] else "-- "} {c["name"]:18s} {c["detail"]}')
    print('reproduced within tolerance' if rep['ok'] else 'NOT reproduced within tolerance')
    return 0 if rep['ok'] else P.FAILED


def cmd_runs(a) -> int:
    from .core.catalog import Catalog
    from .core.paths import tilde, workspace_for
    ws = workspace_for(a.project)
    cat = Catalog(os.path.join(ws, 'catalog.sqlite'))
    if a.rebuild:
        print(f'indexed {cat.rebuild(ws)} runs')
    for r in cat.list('run'):
        s = r['summary']
        print(f'{r["created"]}  {r["status"]:10s} {r["name"][:40]:40s} {tilde(r["path"])}'
              + (f'  {s.get("steps")} steps, {s.get("steps_per_s")} steps/s' if s.get('steps') else ''))
    cat.close()
    return 0


def cmd_gui(a) -> int:
    from .ui.app import main as gui_main
    return gui_main(a.project, fake_carla=a.fake_carla)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog='falcon', description=f'{APP_NAME} {__version__}')
    ap.add_argument('--version', action='version', version=f'{APP_NAME} {__version__}')
    ap.add_argument('--fake-carla', default='', help=argparse.SUPPRESS)   # tests and CI: a fake carla package
    sub = ap.add_subparsers(dest='cmd')
    g = sub.add_parser('gui', help='the application (the default)')
    g.add_argument('project', nargs='?')
    sub.add_parser('doctor', help='check the machine and CARLA')
    pr = sub.add_parser('project', help='create, validate or migrate a project file')
    pr.add_argument('action', choices=('new', 'validate', 'migrate'))
    pr.add_argument('args', nargs='+', help='new: NAME PATH; validate/migrate: PATH')
    for name in ('drive', 'smoke'):
        d = sub.add_parser(name, help='a recorded drive' if name == 'drive' else 'the 30 s smoke drive')
        d.add_argument('project')
        d.add_argument('--out', default='')
        d.add_argument('--seed', type=int, default=0)
        d.add_argument('--keep-server', action='store_true', help='leave Falcon\'s server running afterwards')
        d.add_argument('--max-wall', type=float, default=0.0, help='stop after this many wall-clock seconds')
        if name == 'drive':
            d.add_argument('--world', default='')
            d.add_argument('--rig', default='')
            d.add_argument('--seconds', type=float, default=10.0)
            d.add_argument('--fps', type=float, default=20.0)
    p = sub.add_parser('probe', help='what server answers on Falcon\'s port')
    p.add_argument('--wait', type=float, default=10.0)
    p.add_argument('--keep-server', action='store_true')
    sub.add_parser('tidy', help='tidy Falcon\'s server after a killed worker')
    r = sub.add_parser('reproduce', help='re-run a run and compare within a tolerance')
    r.add_argument('run')
    r.add_argument('--into', default='')
    r.add_argument('--pose-tol', type=float, default=1e-3)
    r.add_argument('--rgb-tol', type=float, default=2.0)
    ru = sub.add_parser('runs', help='list a project\'s runs')
    ru.add_argument('project')
    ru.add_argument('--rebuild', action='store_true')
    sub.add_parser('schema', help='print the project schema\'s path')
    sub.add_parser('export', help='export runs: falcon export RUN... --format F --out DIR (see docs/cli.md)')
    sub.add_parser('evaluate', help='score a registered model on recorded drives, or compare two evaluations '
                                    '(see docs/cli.md)')
    sub.add_parser('gap', help='the sim-to-real gap: falcon gap --sim RUN --real KIND:PATH --out DIR (see docs/cli.md)')
    sub.add_parser('sweep', help='a scenario parameter sweep (see docs/cli.md)')
    sub.add_parser('search', help='a rare-case search over a scenario\'s parameters (see docs/cli.md)')
    return ap


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ['export']:                    # its own parser (core.exporters.main_export)
        from .core.exporters import main_export
        return main_export(argv[1:])
    if argv[:1] == ['evaluate']:                  # the model test harness (core.harness)
        from .core.harness import main_evaluate
        return main_evaluate(argv[1:])
    if argv[:1] == ['gap']:                       # the sim-to-real gap (core.realgap)
        from .core.realgap import main_gap
        return main_gap(argv[1:])
    if argv[:1] in (['sweep'], ['search']):       # scenario studies (sim.studio)
        from .sim.studio import main_study
        return main_study(argv)
    ap = parser()
    a = ap.parse_args(argv)
    if a.cmd is None or a.cmd == 'gui':
        a.project = getattr(a, 'project', None)
        return cmd_gui(a)
    if a.cmd == 'project':
        need = 2 if a.action == 'new' else 1
        if len(a.args) != need:
            ap.error(f'project {a.action} takes {"NAME PATH" if need == 2 else "PATH"}')
        if a.action == 'new':
            a.name, a.path = a.args
        else:
            a.path = a.args[0]
        return cmd_project(a)
    if a.cmd == 'schema':
        from .core.project import SCHEMA_PATH
        print(SCHEMA_PATH)
        return 0
    return {'doctor': cmd_doctor, 'drive': cmd_drive, 'smoke': lambda x: cmd_drive(x, smoke=True),
            'probe': cmd_probe, 'tidy': cmd_tidy, 'reproduce': cmd_reproduce, 'runs': cmd_runs}[a.cmd](a)


if __name__ == '__main__':
    sys.exit(main())
