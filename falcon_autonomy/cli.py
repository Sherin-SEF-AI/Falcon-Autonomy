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
    falcon derive worldmodel|language RUN --out DIR   occupancy, BEV, flow, futures, clips / grounded captions & QA
    falcon model fetch PROJECT ID   download a registered model's pinned weights (SHA-256 checked)
    falcon evaluate PROJECT --model M --run RUN... --out DIR   score a model (or --compare A B)
    falcon gap --sim RUN... --real KIND:PATH --out DIR   the sim-to-real gap of camera images
    falcon train --out DIR --train KIND:PATH... [--val ...] [--arch A --init I]   fine-tune a detector (resumable)
    falcon experiment mixed|list|best|results|export ...   the mixed sim/real experiment, run tracking, MLflow
    falcon translate train|apply|check|experiment|results ...   learned sim-to-real image translation, labels kept
    falcon policy bc|offline|ppo|rollout ...   driving policies: behaviour cloning, PPO, evaluation in the environment
    falcon adversary train|eval|replay ...   learned adversarial traffic: multi-agent PPO, failures, replay
    falcon worldmodel train|eval|imagine|collect|e2e ...   learned world model, imagination, end-to-end driving
    falcon sweep PROJECT --scenario S --name N [--grid K | --samples K]   a parameter sweep (resumable)
    falcon search PROJECT --scenario S --name N [--method bayes|bayes-pof|cmaes|adaptive|random] [--budget B]
                  [--objective criticality|perception|both --model ID --weather FIELD]   rare cases
    falcon stack topics|image|bridge|run|suite|replay|odd|report ...   stack testing: ROS 2 bridge, suites, safety case
    falcon hil device|deploy|session|report|acceptance ...   hardware in the loop: a Jetson (or a simulated device)
    falcon deploy check|build|profile|accuracy|gate|report|all ...   TensorRT engines on the Jetson, profiled and gated
    falcon loop PROJECT [status|next|run --until STEP|show N|new|set]   the closed loop (docs/loop.md)
    falcon report KIND SRC --out FILE          a self-contained HTML report (docs/reports.md)
    falcon replay import|match|run|compare ...   real logs (nuScenes, Waymo v2, ROS 2 bags) replayed with changes
    falcon recon train|from-dashcam|place|eval|render|rerender|info ...   neural reconstruction of a drive (3DGS)
    falcon dashcam list|gps|extract|match ...   a dashcam's clips, GPS track and GPS-stamped frames (docs/dashcam.md)
    falcon dashcam route plan|import|match|drive|pairs ...   drive your dashcam's route in simulation on its streets
    falcon runs search PROJECT|WORKSPACE [QUERY...] [--sort KEY] [--json]   search the runs (docs/runs.md)
    falcon vlm list|verify|fetch|eval|finetune ...   vision-language driving explanations scored against ground truth
    falcon rules check RUN... [--set india|generic|FILE] | list | show SET   traffic rules as temporal logic
    falcon why case|evaluation|episode ...   root-cause attribution of a failure (docs/root-cause.md)
    falcon odd monitor RUN... --odd FILE [--reference REF] | calibrate --in RUN... --out FILE   runtime ODD monitoring
    falcon assets inventory|list|show ID|verify   Indian road assets within the packaged CARLA (docs/indian-assets.md)
    falcon v2x junctions|place|list|remove|drive|benefit|decode|converter ...   roadside units, CPMs (docs/v2x.md)
    falcon ci run CONFIG [--out DIR] | validate | init [TARGET] [--github] | post | lint   CI mode (docs/ci.md)
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
    ru = sub.add_parser('runs', help='list a project\'s runs; falcon runs search PROJECT|WORKSPACE QUERY… searches '
                                     'them (see docs/runs.md)')
    ru.add_argument('project')
    ru.add_argument('--rebuild', action='store_true')
    sub.add_parser('schema', help='print the project schema\'s path')
    sub.add_parser('export', help='export runs: falcon export RUN... --format F --out DIR (see docs/cli.md)')
    sub.add_parser('derive', help='world-model or language data from a drive: falcon derive worldmodel|language RUN '
                                  '--out DIR')
    sub.add_parser('model', help='download a registered model\'s pinned weights: falcon model fetch PROJECT ID')
    sub.add_parser('evaluate', help='score a registered model on recorded drives, or compare two evaluations '
                                    '(see docs/cli.md)')
    sub.add_parser('gap', help='the sim-to-real gap: falcon gap --sim RUN --real KIND:PATH --out DIR (see docs/cli.md)')
    sub.add_parser('odd', help='runtime ODD monitoring: falcon odd monitor|calibrate (see docs/odd-monitoring.md)')
    sub.add_parser('train', help='fine-tune a detector: falcon train --out DIR --train KIND:PATH (see '
                                 'docs/training.md)')
    sub.add_parser('experiment', help='the mixed sim/real training experiment, run tracking and MLflow export (see '
                                      'docs/training.md)')
    sub.add_parser('translate', help='learned sim-to-real image translation, labels kept: falcon translate '
                                     'train|apply|check|experiment|results (see docs/translation.md)')
    sub.add_parser('policy', help='driving policies: falcon policy bc|offline|ppo|rollout (see docs/policies.md)')
    sub.add_parser('adversary', help='learned adversarial traffic: falcon adversary train|eval|replay (see '
                                     'docs/adversarial-traffic.md)')
    sub.add_parser('worldmodel', help='a learned world model, policies learnt in imagination, end-to-end driving: '
                                      'falcon worldmodel train|eval|imagine|collect|e2e (see '
                                      'docs/world-model-training.md)')
    sub.add_parser('sweep', help='a scenario parameter sweep (see docs/cli.md)')
    sub.add_parser('search', help='a rare-case search over a scenario\'s parameters (see docs/cli.md)')
    sub.add_parser('stack', help='stack testing: the ROS 2 bridge, scenario suites, the safety-case report (see '
                                 'docs/stack-testing.md)')
    sub.add_parser('hil', help='hardware in the loop: falcon hil device|deploy|session|report|acceptance (see '
                               'docs/hardware.md)')
    sub.add_parser('deploy', help='TensorRT engines on the Jetson AGX Orin (or the simulated device), profiled and '
                                  'gated: falcon deploy check|build|profile|accuracy|gate|report|all (see '
                                  'docs/deploy.md)')
    sub.add_parser('loop', help='the closed loop: falcon loop PROJECT [status|next|run|show N|new|set] (see '
                                'docs/loop.md)')
    sub.add_parser('report', help='an HTML report: falcon report KIND SRC --out FILE (see docs/reports.md)')
    sub.add_parser('replay', help='log replay: falcon replay import|match|run|compare (see docs/log-replay.md)')
    sub.add_parser('recon', help='neural reconstruction of a drive: falcon recon train|from-dashcam|place|eval|render|'
                                 'rerender|info (see '
                                 'docs/reconstruction.md)')
    sub.add_parser('dashcam', help='a dashcam\'s clips, GPS track and frames: falcon dashcam list|gps|extract|match|'
                                   'route (see docs/dashcam.md)')
    sub.add_parser('vlm', help='vision-language explanations scored against ground truth: falcon vlm '
                               'list|verify|fetch|eval|finetune (see docs/vlm.md)')
    sub.add_parser('rules', help='traffic rules as temporal logic: falcon rules check RUN... | list | show SET (see '
                                 'docs/traffic-rules.md)')
    sub.add_parser('why', help='why did it fail: falcon why case|evaluation|episode (see docs/root-cause.md)')
    sub.add_parser('ci', help='CI mode: a suite headless with gates, JUnit, Markdown and HTML: falcon ci '
                              'run|validate|init|summary|post|lint (see docs/ci.md)')
    sub.add_parser('assets', help='Indian road assets: falcon assets inventory|list|show ID|verify (see '
                                  'docs/indian-assets.md)')
    return ap



def main_derive(argv) -> int:
    """`falcon derive worldmodel|language RUN --out DIR [--option KEY=VALUE]...`: world-model data (core.worldmodel)
    or grounded language data (core.language) from a recorded drive. 0 ok, 2 failed."""
    import argparse
    import json
    ap = argparse.ArgumentParser(prog='falcon derive', description='derive training data from a recorded drive')
    ap.add_argument('what', choices=('worldmodel', 'language'))
    ap.add_argument('run')
    ap.add_argument('--out', '-o', required=True)
    ap.add_argument('--option', action='append', default=[], metavar='KEY=VALUE')
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return 0 if e.code == 0 else 2
    opts = {}
    for kv in a.option:
        k, eq, v = kv.partition('=')
        if not eq:
            print(f'failed: option {kv!r} is not KEY=VALUE', flush=True)
            return 2
        try:
            opts[k] = json.loads(v)
        except ValueError:
            opts[k] = v
    try:
        if a.what == 'worldmodel':
            from .core.worldmodel import derive
            doc = derive(a.run, a.out, opts)
        else:
            from .core.language import derive_language
            doc = derive_language(a.run, a.out, opts)      # no paraphrase client: template text only
    except (OSError, ValueError, KeyError) as e:
        print(f'failed: {e}', flush=True)
        return 2
    skipped = doc.get('skipped') or {}
    print(f'wrote {a.what} data to {a.out}' + (f'; skipped: {json.dumps(skipped)}' if skipped else ''), flush=True)
    return 0

def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ['export']:                    # its own parser (core.exporters.main_export)
        from .core.exporters import main_export
        return main_export(argv[1:])
    if argv[:1] == ['derive']:                    # world-model and language data (core.worldmodel, core.language)
        return main_derive(argv[1:])
    if argv[:1] == ['model']:                     # registered models (core.models)
        from .core.models import main_model
        return main_model(argv[1:])
    if argv[:1] == ['evaluate']:                  # the model test harness (core.harness)
        from .core.harness import main_evaluate
        return main_evaluate(argv[1:])
    if argv[:1] == ['gap']:                       # the sim-to-real gap (core.realgap)
        from .core.realgap import main_gap
        return main_gap(argv[1:])
    if argv[:1] == ['odd']:                       # runtime ODD monitoring (core.oddmonitor)
        from .core.oddmonitor import main_odd
        return main_odd(argv[1:])
    if argv[:1] == ['train']:                     # the training lab (core.training)
        from .core.training import main_train
        return main_train(argv[1:])
    if argv[:1] == ['experiment']:                # experiments: mixed sim/real, tracking, MLflow (core.experiments)
        from .core.experiments import main_experiment
        return main_experiment(argv[1:])
    if argv[:1] == ['translate']:                 # sim-to-real image translation, labels kept (core.translate)
        from .core.translate import main_translate
        return main_translate(argv[1:])
    if argv[:1] == ['policy']:                    # driving policies: BC, PPO, rollouts (core.policy)
        from .core.policy import main_policy
        return main_policy(argv[1:])
    if argv[:1] == ['adversary']:                 # learned adversarial traffic (core.adversary)
        from .core.adversary import main_adversary
        return main_adversary(argv[1:])
    if argv[:1] == ['worldmodel']:                # learned world model, imagination, end to end (core.wmtrain)
        from .core.wmtrain import main_worldmodel
        return main_worldmodel(argv[1:])
    if argv[:1] == ['loop']:                      # the closed loop (core.loop)
        from .core.loop import main_loop
        return main_loop(argv[1:])
    if argv[:1] == ['report']:                    # HTML reports (core.reports)
        from .core.reports import main_report
        return main_report(argv[1:])
    if argv[:1] == ['stack']:                     # stack testing: ROS 2 bridge, suites, safety case (sim.stacksuite)
        from .sim.stacksuite import main_stack
        return main_stack(argv[1:])
    if argv[:1] == ['hil']:                       # hardware in the loop (falcon_autonomy.hil)
        from .hil.cli import main_hil
        return main_hil(argv[1:])
    if argv[:1] == ['deploy']:                    # TensorRT engines on the Jetson, profiled and gated (core.deploy)
        from .core.deploy import main_deploy
        return main_deploy(argv[1:])
    if argv[:1] == ['rules']:                     # traffic rules as temporal logic (core.rules)
        from .core.rules import main_rules
        return main_rules(argv[1:])
    if argv[:1] == ['ci']:                        # CI mode: suites headless, gates, JUnit and Markdown (core.ci)
        from .core.ci import main_ci
        return main_ci(argv[1:])
    if argv[:1] == ['v2x']:                       # cooperative roadside units and their CPMs (core.v2x)
        from .core.v2x import main_v2x
        return main_v2x(argv[1:])
    if argv[:1] == ['assets']:                    # Indian road assets within the packaged CARLA (core.assets)
        from .core.assets import main_assets
        return main_assets(argv[1:])
    if argv[:1] == ['why']:                       # root-cause attribution (core.rootcause)
        from .core.rootcause import main_why
        return main_why(argv[1:])
    if argv[:1] == ['replay']:                    # log replay: import, re-simulate with changes, compare
        from .core.logreplay.commands import main_replay
        return main_replay(argv[1:])
    if argv[:1] == ['recon']:                     # neural reconstruction of a drive (core.recon)
        from .core.recon.commands import main_recon
        return main_recon(argv[1:])
    if argv[:2] == ['dashcam', 'route']:          # drive a dashcam route in simulation (core.dashcam_route)
        from .core.dashcam_route import main_route
        return main_route(argv[2:])
    if argv[:1] == ['dashcam']:                   # a dashcam's clips, GPS track and frames (core.dashcam)
        from .core.dashcam import main_dashcam
        return main_dashcam(argv[1:])
    if argv[:2] == ['runs', 'search']:            # search a workspace's runs (core.runindex)
        from .core.runindex import main_runs
        return main_runs(argv[2:])
    if argv[:1] == ['vlm']:                       # vision-language explanations scored against the truth (core.vlm)
        from .core.vlm_eval import main_vlm
        return main_vlm(argv[1:])
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
