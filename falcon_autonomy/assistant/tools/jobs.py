"""Job tools: queue what the pages queue — drives, scenario runs, sweeps and rare-case searches, training, evaluation,
exports, reconstruction, translation, stack suites, V2X benefit, derived data, reports, root cause, the loop,
reproduction — and follow, cancel and wait for jobs.

Each acting tool builds the same config or command line as its page (core.drive.drive_config and
scenario_config through CarlaService.submit_drive; `python -m falcon_autonomy …` through the job queue with the page's
JobSpec, using the factored builders sim.studio.sweep_argv/search_argv, core.exporters.export_argv,
core.stack.suite_run_argv and the core job_spec helpers), so a job the assistant starts is the job the user would
have started: the same locks, memory guard, outputs and history. Its estimate comes from this machine's measured
throughput (core.budget) where the job's cost is a rendering cost, else None with what it depends on; its preview is
the command line (`falcon …`) or the drive's settings.

Inputs are ids (runs, worlds, rigs, scenarios, models, studies, suites, translators, reconstructions); outputs go into
their usual workspace folders. A job that reads a project file (sweeps, searches, suites, the loop, why) saves the
project first, as the pages do.
"""
from __future__ import annotations

import os
import threading
import time

from ..tool_api import ToolError, tool
from . import _common as C

WAIT_MAX_S = 600.0


# ------------------------------------------------------------------------------------------------ helpers
def _fake(ctx) -> str:
    svc = ctx.service if ctx.win is None else ctx.win.service
    return getattr(svc, 'fake_carla', '') or ''


def _saved(ctx) -> str:
    """The project's file, saved (jobs that read the project read the file, as the pages make sure)."""
    p = C.project(ctx)
    if not p.path:
        raise ToolError('the project has no file yet: save it (File ▸ Save as) before this job')
    if p.dirty:
        if ctx.win is not None:
            if not ctx.win.save():
                raise ToolError('the project could not be saved')
        else:
            p.save()
    return p.path


def _stamp() -> str:
    return time.strftime('%Y%m%d-%H%M%S')


def _new_dir(ctx, sub: str, name: str) -> str:
    """<ws>/<sub>/<name> made now (-2, -3 … when taken), as the pages' new_dir."""
    base = os.path.join(C.workspace(ctx), sub, name)
    p, k = base, 2
    while os.path.exists(p):
        p, k = f'{base}-{k}', k + 1
    os.makedirs(p)
    return p


def _world_rig(ctx, world, rig):
    d = C.doc(ctx)
    w = C.find(ctx, 'worlds', world, 'world')[1] if world else d['worlds'][0]
    r = C.find(ctx, 'rigs', rig, 'rig')[1] if rig else d['rigs'][0]
    return w, r


def _drive_estimate(ctx, rig: dict, steps: int, n: int = 1, setup_s: float = 20.0) -> dict:
    from ...core import budget
    t = C.need_service(ctx).throughput()
    est = t.estimate(int(steps), budget.rig_mpx(rig), setup_s)
    what = f'{n} × {steps} steps at {budget.rig_mpx(rig):.1f} Mpx per step, {est["label"]} ({C.clip(est["source"], 120)})'
    return {'seconds': round(est['seconds'] * n, 1), 'what': what}


def _scenario(ctx, ident):
    from ...core import scenarios as SC
    if ident in SC.LIBRARY:
        return SC.builtin(ident)
    return C.find(ctx, 'scenarios', ident, 'scenario')[1]


def _runs(ctx, runs) -> list:
    if not runs:
        raise ToolError('give at least one run id')
    return [C.resolve_run(ctx, r) for r in runs]


def _model(ctx, mid):
    return C.find(ctx, 'models', mid, 'model')[1]


def _cli_preview(argv_fn):
    def pv(ctx, **a):
        _kind, _title, argv, _kw = argv_fn(ctx, dry=True, **a)
        return C.command_line(argv)
    return pv


def _queue(ctx, built) -> dict:
    kind, title, argv, kw = built
    return C.cli_job(ctx, kind, title, argv, **kw)


# ================================================================================================= drives
def _drive_cfg(ctx, world='', rig='', seconds=10.0, fps=20.0, seed=0, weather_preset='', triggers='all', name=''):
    from ...core import weather as W
    from ...core.drive import drive_config, run_dir_name
    from ...core.project import ProjectError
    from ...core.triggers import attach
    svc = C.need_service(ctx)
    w, r = _world_rig(ctx, world, rig)
    out = os.path.join(C.workspace(ctx), 'runs', run_dir_name(name or f'drive-{w["name"]}'))
    try:
        cfg = drive_config(C.project(ctx), w['id'], r['id'], out, svc.port, svc.tm_port, float(seconds), float(fps),
                           int(seed), render_budget_mpx=float(svc.settings.get('render_budget_mpx')),
                           workspace=C.workspace(ctx))
        if weather_preset:
            if weather_preset not in W.PRESETS:
                raise ToolError(f'no weather preset {weather_preset!r}')
            cfg['weather'] = W.resolve({'preset': weather_preset})
            cfg['weather_source'] = {'preset': weather_preset, 'override': 'the assistant (as Data ▸ Record)'}
        if triggers != 'none':
            attach(cfg, C.project(ctx), 'drives', None if triggers == 'all' else [x.strip() for x in
                                                                                    triggers.split(',')])
    except ProjectError as e:
        raise ToolError(str(e)) from e
    from ...core import budget
    ok, why = budget.check_budget(r, float(svc.settings.get('render_budget_mpx')))
    if not ok:
        raise ToolError(f'refused before it starts: {why}')
    return cfg, r


_DRIVE = C.obj({'world': C.s('world id (default the first)'), 'rig': C.s('rig id (default the first)'),
                'seconds': C.n('simulated seconds (default 10)', minimum=0.5, maximum=3600),
                'fps': C.n('steps per second (default 20)', minimum=1, maximum=100),
                'seed': C.i('seed (default 0): the same seed repeats the run'),
                'weather_preset': C.s('a weather preset for this drive only'),
                'triggers': C.s('trigger rules: all (the project\'s enabled rules for drives), none, or ids a,b'),
                'name': C.s('a label for the run folder')})


def _drive_preview(ctx, **a):
    cfg, r = _drive_cfg(ctx, **a)
    return (f'record a drive: {cfg["name"]}, {cfg["steps"]} steps at {cfg["fps"]:g} Hz, seed {cfg["seed"]}, '
            f'weather {cfg["weather_source"].get("preset", "custom")}, {cfg["traffic"].get("vehicles", 0)} vehicles, '
            f'{len((cfg.get("trigger_rules") or {}).get("rules") or [])} trigger rule(s) → runs/'
            f'{os.path.basename(cfg["out"])}')


@tool('jobs_drive', 'jobs', 'act',
      'Record a drive (Data ▸ Record): the world and rig, length, step, seed, an optional weather preset, the '
      'project\'s trigger rules for drives; the ego and traffic on the traffic manager; every sensor and label the '
      'rig asks for. Queued behind the machine-wide CARLA lock (Falcon\'s own server is started for it). Returns the '
      'job id and the run id it will write.', _DRIVE,
      estimate=lambda ctx, **a: _drive_estimate(ctx, _drive_cfg(ctx, **a)[1],
                                                int(round(float(a.get('seconds', 10)) * float(a.get('fps', 20))))),
      preview=_drive_preview)
def jobs_drive(ctx, **a):
    cfg, _r = _drive_cfg(ctx, **a)
    try:
        job = C.need_service(ctx).submit_drive(cfg)
    except RuntimeError as e:
        raise ToolError(str(e)) from e
    ctx.emit('job_queued', {'job': job.id, 'title': job.spec.title})
    return {'job': job.id, 'title': job.spec.title, 'run': C.run_id(ctx, cfg['out']),
            'follow': 'jobs_wait, then runs_get with the run id'}


def _scn_cfg(ctx, scenario, world='', rig='', seed=0, values=None, fps=20.0):
    from ...core.drive import run_dir_name, scenario_config
    from ...core.project import ProjectError
    svc = C.need_service(ctx)
    s = _scenario(ctx, scenario)
    w, r = _world_rig(ctx, world, rig)
    out = os.path.join(C.workspace(ctx), 'runs', run_dir_name(f'scenario-{s["id"]}'))
    try:
        cfg = scenario_config(C.project(ctx), s, w['id'], r['id'], out, svc.port, svc.tm_port, seed=int(seed),
                              values=values or None, fps=float(fps),
                              render_budget_mpx=float(svc.settings.get('render_budget_mpx')),
                              workspace=C.workspace(ctx), trigger_rules='project')
    except (ProjectError, ValueError, KeyError) as e:
        raise ToolError(str(e)) from e
    return cfg, s, r


_SCN = C.obj({'scenario': C.s('a built-in key or project scenario id'), 'world': C.s('world id'),
              'rig': C.s('rig id'), 'seed': C.i('seed (samples the parameters; default 0)'),
              'values': {'type': 'object', 'description': 'parameter values to use instead of sampling (all of them, '
                                                          'or the rest are sampled)'},
              'fps': C.n('steps per second (default 20)', minimum=1, maximum=100)}, ['scenario'])


@tool('jobs_scenario_run', 'jobs', 'act',
      'Run a scenario once (Scenarios ▸ Run): sampled with the seed (or the given parameter values), the world '
      'reloaded so it repeats bit for bit, background traffic off, the project\'s trigger rules for scenarios. The '
      'run records its checks (pass/fail) and criticality (TTC, PET, distance, deceleration).', _SCN,
      estimate=lambda ctx, **a: (lambda c: _drive_estimate(ctx, c[2], c[0]['steps']))(_scn_cfg(ctx, **a)),
      preview=lambda ctx, **a: (lambda c: f'run {c[1]["name"]} once: seed {c[0]["seed"]}, values '
                                          f'{C._short((c[0].get("scenario") or {}).get("values"), 200)}, '
                                          f'{c[0]["steps"]} steps → runs/{os.path.basename(c[0]["out"])}')(
          _scn_cfg(ctx, **a)))
def jobs_scenario_run(ctx, **a):
    cfg, s, _r = _scn_cfg(ctx, **a)
    try:
        job = C.need_service(ctx).submit_drive(cfg, title=f'{s["name"]} (seed {cfg["seed"]})')
    except RuntimeError as e:
        raise ToolError(str(e)) from e
    ctx.emit('job_queued', {'job': job.id, 'title': job.spec.title})
    return {'job': job.id, 'title': job.spec.title, 'run': C.run_id(ctx, cfg['out'])}


def _sweep(ctx, scenario, samples=20, seed=0, world='', rig='', name='', dry=False):
    from ...sim.studio import sweep_argv
    s = _scenario(ctx, scenario)
    w, r = _world_rig(ctx, world, rig)
    path = C.project(ctx).path if dry else _saved(ctx)
    nm = name or f'sweep-{s["id"]}'
    argv = sweep_argv(path or '<the project file>', s['id'], nm, int(samples), int(seed), w['id'], r['id'], _fake(ctx))
    return 'study', f'Sweep {s["name"]} ({samples})', argv, {'memory_gb': 0.5}


@tool('jobs_sweep', 'jobs', 'act',
      'A scenario parameter sweep (Scenarios ▸ Sweep): N samples of the parameters, each a scenario run in the '
      'study studies/<name> (resumable: re-running continues it). Analyse with analysis_study.',
      C.obj({'scenario': C.s('scenario key or id'), 'samples': C.i('runs (default 20)', minimum=2, maximum=10000),
             'seed': C.i('seed'), 'world': C.s('world id'), 'rig': C.s('rig id'),
             'name': C.s('study name (default sweep-<scenario>)')}, ['scenario']),
      estimate=lambda ctx, **a: _drive_estimate(ctx, _world_rig(ctx, a.get('world'), a.get('rig'))[1],
                                                int(float(_scenario(ctx, a['scenario']).get('duration_s', 12)) * 20),
                                                int(a.get('samples', 20))),
      preview=_cli_preview(_sweep))
def jobs_sweep(ctx, **a):
    return _queue(ctx, _sweep(ctx, **a))


def _search(ctx, scenario, method='bayes', budget=60, batch=4, threshold=0.5, seed=0, world='', rig='',
            objective='criticality', count='avoidable', feasibility=True, model='', weather=(), fps=20, dry=False):
    from ...sim.studio import search_argv, search_name
    s = _scenario(ctx, scenario)
    w, r = _world_rig(ctx, world, rig)
    if objective != 'criticality':
        if not model:
            raise ToolError('the perception objective needs a registered model (project_overview lists them)')
        _model(ctx, model)
    path = C.project(ctx).path if dry else _saved(ctx)
    nm = search_name(s['id'], objective, model, list(weather), int(fps))
    argv = search_argv(path or '<the project file>', s['id'], nm, method, int(budget), int(batch), float(threshold),
                       int(seed), w['id'], r['id'], objective, count, bool(feasibility), model, list(weather),
                       int(fps), _fake(ctx))
    title = f'Search {s["name"]} ({method}, {budget}' + (f', {objective}' if objective != 'criticality' else '') + ')'
    return 'study', title, argv, {'memory_gb': 0.5}


@tool('jobs_search', 'jobs', 'act',
      'A rare-case search (Rare cases): Bayesian optimisation (bayes, bayes-pof), CMA-ES, adaptive or random '
      'sampling over a scenario\'s parameters (and optionally fog, sun, rain), for criticality (by default counting '
      'only failures the ego could have avoided by RSS, over the kinematically feasible parameters) or a model\'s '
      'perception misses. Resumable; the study is studies/<name> (returned). Analyse with analysis_study.',
      C.obj({'scenario': C.s('scenario key or id'),
             'method': C.s('search method', enum=['bayes', 'bayes-pof', 'cmaes', 'adaptive', 'random']),
             'budget': C.i('runs (default 60)', minimum=4, maximum=5000), 'batch': C.i('runs per batch (default 4)',
                                                                                     minimum=1, maximum=64),
             'threshold': C.n('failure threshold of the objective (default 0.5)', minimum=0, maximum=1),
             'seed': C.i('seed'), 'world': C.s('world id'), 'rig': C.s('rig id'),
             'objective': C.s('what to search for', enum=['criticality', 'perception', 'both']),
             'count': C.s('criticality: count avoidable failures only (RSS) or all', enum=['avoidable', 'all']),
             'feasibility': C.b('restrict to kinematically feasible parameters (default true)'),
             'model': C.s('a registered model id (perception, both)'),
             'weather': C.arr(C.s('field', enum=['fog_density', 'sun_altitude_angle', 'precipitation']),
                              'weather fields to search too'),
             'fps': C.i('steps per second (default 20)', minimum=5, maximum=100)}, ['scenario']),
      estimate=lambda ctx, **a: _drive_estimate(ctx, _world_rig(ctx, a.get('world'), a.get('rig'))[1],
                                                int(float(_scenario(ctx, a['scenario']).get('duration_s', 12))
                                                    * float(a.get('fps', 20))), int(a.get('budget', 60))),
      preview=_cli_preview(_search))
def jobs_search(ctx, **a):
    out = _queue(ctx, _search(ctx, **a))
    out['study'] = out['command'].split('--name ', 1)[1].split(' ', 1)[0]
    return out


@tool('jobs_replay_failure', 'jobs', 'act',
      'Replay one evaluation of a study (a sweep or search) as a scenario run with its exact parameter values and '
      'weather (Rare cases ▸ Replay), to look at it in Data ▸ Browse.',
      C.obj({'study': C.s('study id (folder name in studies/)'), 'index': C.i('the row index from analysis_study',
                                                                                minimum=0),
             'seed': C.i('seed (default the study\'s)')}, ['study', 'index']),
      preview=lambda ctx, **a: f'replay row {a["index"]} of studies/{a["study"]} as a scenario run')
def jobs_replay_failure(ctx, study, index, seed=None):
    from ...core import milsearch as MS
    from ...core.drive import run_dir_name, scenario_config
    from ...core.project import ProjectError
    d = C.resolve_in(ctx, 'studies', study, 'study', ('study.json',))
    st = C.read_json(os.path.join(d, 'study.json'), {})
    rows = C.read_jsonl(os.path.join(d, 'results.jsonl'))
    if not 0 <= index < len(rows):
        raise ToolError(f'the study has {len(rows)} rows')
    svc = C.need_service(ctx)
    scn = st['scenario'] if isinstance(st.get('scenario'), dict) else _scenario(ctx, st.get('scenario'))
    values, weather = MS.split_values(rows[index].get('values') or {})
    out = os.path.join(C.workspace(ctx), 'runs', run_dir_name('replay'))
    try:
        cfg = scenario_config(C.project(ctx), scn, st.get('world'), st.get('rig'), out, svc.port, svc.tm_port,
                              seed=int(st.get('seed', 0) if seed is None else seed), values=values,
                              workspace=C.workspace(ctx), trigger_rules='project')
    except (ProjectError, ValueError, KeyError) as e:
        raise ToolError(str(e)) from e
    MS.apply_weather(cfg, weather)
    job = svc.submit_drive(cfg, title=f'Replay: {scn["name"]}')
    return {'job': job.id, 'title': job.spec.title, 'run': C.run_id(ctx, out)}


@tool('jobs_reproduce', 'jobs', 'act',
      'Re-run a recorded drive from its own config (drive.json) and compare it with the original within a tolerance '
      '(steps, ego trajectory, images, labels, points): `falcon reproduce`. The reproduction is <run>-reproduced with '
      'reproduce.json (its checks); the job fails when it does not agree.',
      C.obj({'run': C.s('a run id (a drive)')}, ['run']),
      preview=lambda ctx, run: C.command_line(['reproduce', C.resolve_run(ctx, run)]))
def jobs_reproduce(ctx, run):
    d = C.resolve_run(ctx, run)
    if not os.path.isfile(os.path.join(d, 'drive.json')):
        raise ToolError('only drives (with drive.json) can be reproduced')
    if os.path.exists(d.rstrip('/') + '-reproduced'):
        raise ToolError(f'{C.run_id(ctx, d)}-reproduced exists already')
    argv = (['--fake-carla', _fake(ctx)] if _fake(ctx) else []) + ['reproduce', d]     # a top-level option there
    out = C.cli_job(ctx, 'reproduce', f'Reproduce {os.path.basename(d)}', argv, memory_gb=0.5,
                    run_dir=d.rstrip('/') + '-reproduced')
    out['run'] = C.run_id(ctx, d) + '-reproduced'
    return out


# ================================================================================================= training
def _train_detector(ctx, runs, val='', arch='fasterrcnn_mobilenet_v3_large_fpn', init='coco', epochs=12, batch=8,
                    size='640x384', seed=0, dry=False):
    from ...core import training as T
    if arch not in T.ARCHS:
        raise ToolError(f'arch must be one of {", ".join(T.ARCHS)}')
    tid = f'{_stamp()}-{arch}'
    argv = ['--workspace', C.workspace(ctx), '--id', tid, '--arch', arch, '--init', init, '--epochs', str(epochs),
            '--batch', str(batch), '--size', size, '--seed', str(seed)]
    for r in _runs(ctx, runs):
        argv += ['--train', f'falcon:{r}']
    if val:
        argv += ['--val', f'falcon:{C.resolve_run(ctx, val)}']
    return 'train', f'Train {arch} on {len(runs)} drive(s)', ['train'] + argv, {
        'spec': None if dry else T.job_spec(argv, f'Train {arch} on {len(runs)} drive(s)')}


@tool('jobs_train_detector', 'jobs', 'act',
      'Fine-tune a torchvision detector on recorded drives (Training ▸ Detectors): seeded, resumable, exported to '
      'ONNX and checked, registered as a model; a GPU job (Falcon\'s idle server is stopped for it). Pretrained '
      'weights must already be fetched by the user (licence acceptance is theirs). Output training/<id>.',
      C.obj({'runs': C.arr(C.s('run id'), 'training drives', minItems=1, maxItems=500),
             'val': C.s('a validation run id'), 'arch': C.s('architecture (core.training.ARCHS)'),
             'init': C.s('initial weights', enum=['coco', 'imagenet', 'scratch']),
             'epochs': C.i('epochs (default 12)', minimum=1, maximum=500),
             'batch': C.i('batch size (default 8)', minimum=1, maximum=128),
             'size': C.s('input WxH (default 640x384)', pattern=r'^\d+x\d+$'), 'seed': C.i('seed')}, ['runs']),
      estimate=lambda ctx, **a: {'seconds': None, 'what': 'GPU time: depends on frames × epochs; the job reports '
                                                          'its ETA once running'},
      preview=_cli_preview(_train_detector))
def jobs_train_detector(ctx, **a):
    return _queue(ctx, _train_detector(ctx, **a))


def _train_policy(ctx, kind, runs=None, updates=50, steps=2048, dry=False):
    dirs = _runs(ctx, runs) if kind == 'bc' else []
    out = os.path.join(C.workspace(ctx), 'policies', f'{kind}-{_stamp()}') if dry else _new_dir(
        ctx, 'policies', f'{kind}-{_stamp()}')
    if kind == 'bc':
        argv = ['policy', 'bc', '--out', out, '--epochs', '30']
        for r in dirs:
            argv += ['--run', r]
        return 'policy', f'Behaviour cloning on {len(runs)} drive(s)', argv, {'memory_gb': 2.0}
    path = C.project(ctx).path if dry else _saved(ctx)
    argv = ['policy', 'ppo', path or '<the project file>', '--out', out, '--curriculum', 'default', '--updates',
            str(updates), '--steps', str(steps)] + (['--fake-carla', _fake(ctx)] if _fake(ctx) else [])
    return 'policy', f'PPO, {updates} updates', argv, {'memory_gb': 2.0}


@tool('jobs_train_policy', 'jobs', 'act',
      'Train a driving policy (Training ▸ Driving policies): behaviour cloning from recorded drives (bc), or PPO '
      'through the CARLA environment with the default scenario curriculum (ppo). Output policies/<kind>-<stamp>; '
      'score it with the Leaderboard on the Training page.',
      C.obj({'kind': C.s('bc or ppo', enum=['bc', 'ppo']), 'runs': C.arr(C.s('run id'), 'bc: the drives'),
             'updates': C.i('ppo updates (default 50)', minimum=1, maximum=10000),
             'steps': C.i('ppo steps per update (default 2048)', minimum=64, maximum=100000)}, ['kind']),
      estimate=lambda ctx, **a: {'seconds': None, 'what': 'bc: minutes on the CPU/GPU; ppo: CARLA steps × updates'},
      preview=_cli_preview(_train_policy))
def jobs_train_policy(ctx, **a):
    return _queue(ctx, _train_policy(ctx, **a))


def _real_source(ctx, real: str) -> str:
    """KIND:NAME of a real image set inside the workspace (realsets/<name> or a folder there), checked."""
    if ':' not in real:
        raise ToolError('real is KIND:FOLDER, e.g. kitti:realsets/kitti-val (a folder inside the workspace)')
    kind, rel = real.split(':', 1)
    if kind == 'dashcam':
        raise ToolError('private: the dashcam footage never reaches the assistant; start that from the window')
    if C._bad_name(rel):
        raise ToolError(f'{rel!r} is not a folder in the workspace')
    p = os.path.join(C.workspace(ctx), rel)
    if not os.path.exists(p):
        raise ToolError(f'no {rel} in the workspace (a real image set is used from inside it)')
    return f'{kind}:{C.checked(ctx, p)}'


def _train_translator(ctx, runs, real, iterations=20000, seed=0, dry=False):
    from ...core import translate as TR
    tid = f'{_stamp()}-cut'
    argv = ['train', '--workspace', C.workspace(ctx), '--id', tid, '--real', _real_source(ctx, real),
            '--iterations', str(iterations), '--seed', str(seed)]
    for r in _runs(ctx, runs):
        argv += ['--sim', r]
    return 'translate', f'Train translator {tid}', ['translate'] + argv, {
        'spec': None if dry else TR.job_spec(argv, f'Train translator {tid}')}


@tool('jobs_train_translator', 'jobs', 'act',
      'Train a sim-to-real image translator (CUT, labels kept) on drives against a real image set inside the '
      'workspace (Training ▸ Sim-to-real translation): a GPU job, resumable. Output translators/<id>; apply it with '
      'jobs_translate_apply.',
      C.obj({'runs': C.arr(C.s('run id'), 'simulated drives', minItems=1, maxItems=200),
             'real': C.s('KIND:FOLDER of a real set inside the workspace (kitti, nuscenes, bdd, coco, folder)'),
             'iterations': C.i('iterations (default 20000)', minimum=100, maximum=1000000), 'seed': C.i('seed')},
            ['runs', 'real']),
      estimate=lambda ctx, **a: {'seconds': None, 'what': 'GPU time ∝ iterations; ETA reported once running'},
      preview=_cli_preview(_train_translator))
def jobs_train_translator(ctx, **a):
    return _queue(ctx, _train_translator(ctx, **a))


def _translate_apply(ctx, translator, runs, threshold=None, on_drift='flag', pin_actors=True, dry=False):
    from ...core import translate as TR
    model = C.resolve_in(ctx, 'translators', translator, 'translator')
    jobs = []
    for r in _runs(ctx, runs):
        out = os.path.join(C.workspace(ctx), 'runs', TR.default_name(r, model))
        if os.path.exists(out):
            raise ToolError(f'runs/{os.path.basename(out)} exists already')
        argv = ['apply', '--model', model, '--run', r, '--out', out, '--on-drift', on_drift]
        if threshold is not None:
            argv += ['--threshold', f'{threshold:g}']
        if not pin_actors:
            argv.append('--no-pin-actors')
        jobs.append(('translate', f'Translate {os.path.basename(r)}', ['translate'] + argv,
                     {'spec': None if dry else TR.job_spec(argv, f'Translate {os.path.basename(r)}', memory_gb=3.0,
                                                           run_dir=out)}))
    return jobs


@tool('jobs_translate_apply', 'jobs', 'act',
      'Translate drives with a trained translator into new runs (runs/<run>-tr-<translator>) whose labels are kept, '
      'each frame\'s drift measured (frames over the threshold flagged or left raw). One GPU job per run.',
      C.obj({'translator': C.s('translator id (folder in translators/)'),
             'runs': C.arr(C.s('run id'), 'drives', minItems=1, maxItems=50),
             'threshold': C.n('drift threshold (default the translator\'s)'),
             'on_drift': C.s('flag or raw', enum=['flag', 'raw']),
             'pin_actors': C.b('keep actors pinned (default true)')}, ['translator', 'runs']),
      preview=lambda ctx, **a: '\n'.join(C.command_line(j[2]) for j in _translate_apply(ctx, dry=True, **a)))
def jobs_translate_apply(ctx, **a):
    return {'jobs': [_queue(ctx, j) for j in _translate_apply(ctx, **a)]}


def _train_worldmodel(ctx, runs, iterations=2000, seed=0, device='auto', dry=False):
    from ...core import wmtrain as WT
    mid = f'{_stamp()}-rssm'
    argv = ['train', '--workspace', C.workspace(ctx), '--id', mid, '--iterations', str(iterations), '--seed',
            str(seed), '--device', device]
    for r in _runs(ctx, runs):
        argv += ['--run', r]
    gpu = device != 'cpu'
    return 'worldmodel', f'World model {mid}', ['worldmodel'] + argv, {
        'spec': None if dry else WT.job_spec(argv, f'World model {mid}', needs_gpu=gpu)}


@tool('jobs_train_worldmodel', 'jobs', 'act',
      'Train a latent world model (RSSM) on drives (Training ▸ World model), scored against copy-last and '
      'constant-velocity baselines per horizon. Output worldmodels/<id>.',
      C.obj({'runs': C.arr(C.s('run id'), 'drives', minItems=1, maxItems=500),
             'iterations': C.i('iterations (default 2000)', minimum=10, maximum=1000000), 'seed': C.i('seed'),
             'device': C.s('auto, cpu or cuda', enum=['auto', 'cpu', 'cuda'])}, ['runs']),
      estimate=lambda ctx, **a: {'seconds': None, 'what': 'GPU time ∝ iterations'},
      preview=_cli_preview(_train_worldmodel))
def jobs_train_worldmodel(ctx, **a):
    return _queue(ctx, _train_worldmodel(ctx, **a))


# ================================================================================================= evaluation
def _evaluate(ctx, model, runs, every=2, boot=1000, seed=0, dry=False):
    m = _model(ctx, model)
    dirs = _runs(ctx, runs)
    path = C.project(ctx).path if dry else _saved(ctx)
    out = os.path.join(C.workspace(ctx), 'evaluations', f'{_stamp()}-{m["id"]}') if dry else _new_dir(
        ctx, 'evaluations', f'{_stamp()}-{m["id"]}')
    argv = ['evaluate', path or '<the project file>', '--model', m['id'], '--out', out, '--every', str(every),
            '--boot', str(boot), '--seed', str(seed)]
    for r in dirs:
        argv += ['--run', r]
    return 'evaluate', f'Evaluate {m.get("name")} on {len(runs)} drive(s)', argv, {
        'needs_gpu': m.get('runtime') != 'python', 'memory_gb': 2.0}


@tool('jobs_evaluate', 'jobs', 'act',
      'Score a registered model on recorded drives (Models ▸ Evaluate): COCO AP/AR or mIoU with drive-level '
      'bootstrap intervals, breakdowns by range, occlusion and size, a condition grid and the worst frames. Output '
      'evaluations/<id>; read with analysis_evaluation. Weights must already be fetched by the user.',
      C.obj({'model': C.s('model id'), 'runs': C.arr(C.s('run id'), 'drives', minItems=1, maxItems=500),
             'every': C.i('every Nth frame (default 2)', minimum=1, maximum=100),
             'boot': C.i('bootstrap resamples (default 1000)', minimum=0, maximum=100000), 'seed': C.i('seed')},
            ['model', 'runs']),
      estimate=lambda ctx, **a: {'seconds': None, 'what': 'inference over the drives\' frames / every'},
      preview=_cli_preview(_evaluate))
def jobs_evaluate(ctx, **a):
    return _queue(ctx, _evaluate(ctx, **a))


def _compare_evals(ctx, a, b, boot=1000, dry=False):
    da = C.resolve_in(ctx, 'evaluations', a, 'evaluation', ('evaluation.json',))
    db = C.resolve_in(ctx, 'evaluations', b, 'evaluation', ('evaluation.json',))
    name = f'{os.path.basename(da)}-vs-{os.path.basename(db)}'
    out = os.path.join(C.workspace(ctx), 'comparisons', name) if dry else _new_dir(ctx, 'comparisons', name)
    return 'compare', f'Compare {a} and {b}', ['evaluate', '--compare', da, db, '--out', out, '--boot', str(boot)], {}


@tool('jobs_compare_evaluations', 'jobs', 'act',
      'A paired bootstrap of every number\'s change B − A between two evaluations (Models ▸ Comparison): '
      'regressions and improvements with intervals. Output comparisons/<a>-vs-<b>.',
      C.obj({'a': C.s('evaluation id (folder in evaluations/)'), 'b': C.s('evaluation id'),
             'boot': C.i('resamples (default 1000)', minimum=100, maximum=100000)}, ['a', 'b']),
      preview=_cli_preview(_compare_evals))
def jobs_compare_evaluations(ctx, **a):
    return _queue(ctx, _compare_evals(ctx, **a))


def _gap(ctx, runs, real, images=500, features=False, match=False, dry=False):
    dirs, src = _runs(ctx, runs), _real_source(ctx, real)
    out = os.path.join(C.workspace(ctx), 'gaps', _stamp()) if dry else _new_dir(ctx, 'gaps', _stamp())
    argv = ['gap', '--real', src, '--out', out, '--images', str(images), '--workspace',
            C.workspace(ctx)]
    for r in dirs:
        argv += ['--sim', r]
    if features:
        argv.append('--features')
    if match:
        argv.append('--match')
    return 'gap', f'Sim-to-real gap of {len(runs)} drive(s)', argv, {'needs_gpu': bool(features), 'memory_gb': 3.0}


@tool('jobs_gap', 'jobs', 'act',
      'Measure the sim-to-real gap of drives against a real image set inside the workspace (Models ▸ Sim-to-real): '
      'appearance statistics, and with features true Fréchet/kernel distances next to their real-vs-real floor '
      '(the feature extractor must be fetched by the user). Output gaps/<stamp>; read with analysis_gap.',
      C.obj({'runs': C.arr(C.s('run id'), 'drives', minItems=1, maxItems=200),
             'real': C.s('KIND:FOLDER of a real set inside the workspace'),
             'images': C.i('images per side (default 500)', minimum=10, maximum=50000),
             'features': C.b('feature distances (GPU)'), 'match': C.b('measured appearance matching')},
            ['runs', 'real']),
      preview=_cli_preview(_gap))
def jobs_gap(ctx, **a):
    return _queue(ctx, _gap(ctx, **a))


# ================================================================================================= data products
def _export(ctx, run, format, name='', dry=False):
    from ...core.exporters import CHOICES, export_argv
    if format not in CHOICES:
        raise ToolError(f'format must be one of {", ".join(CHOICES)}')
    d = C.resolve_run(ctx, run)
    nm = name or f'{os.path.basename(d)}-{format}'
    out = os.path.join(C.workspace(ctx), 'exports', nm)
    if not dry and os.path.exists(out):
        raise ToolError(f'exports/{nm} exists already: give another name')
    return 'export', f'Export {os.path.basename(d)} as {format}', export_argv(d, format, out), {'memory_gb': 1.0}


@tool('jobs_export', 'jobs', 'act',
      'Export a run as COCO, YOLO, KITTI, nuScenes (lidarseg, panoptic), Waymo-style JSON, BDD lanes, ROS 2 MCAP or '
      'semantic occupancy (occ3d, openocc, surroundocc) into exports/<name> (Data ▸ Browse ▸ Export); '
      'falcon_export.json there says what was filled, left out and why.',
      C.obj({'run': C.s('run id'), 'format': C.s('format', enum=['coco', 'yolo', 'kitti', 'nuscenes', 'waymo',
                                                                  'bdd_lanes', 'mcap', 'occ3d', 'openocc',
                                                                  'surroundocc']),
             'name': C.s('folder name under exports/')}, ['run', 'format']),
      preview=_cli_preview(_export))
def jobs_export(ctx, **a):
    return _queue(ctx, _export(ctx, **a))


def _derive(ctx, what, run, dry=False):
    d = C.resolve_run(ctx, run)
    if what == 'audio':
        if not os.path.isfile(os.path.join(d, 'actors.jsonl')):
            raise ToolError('synthetic audio needs the drive\'s actors.jsonl')
        return 'derive', f'Synthetic audio of {os.path.basename(d)}', ['derive', 'audio', d], {'memory_gb': 2.0}
    out = os.path.join(C.workspace(ctx), 'derived', f'{os.path.basename(d)}-{what}')
    if not dry and os.path.exists(out):
        raise ToolError(f'derived/{os.path.basename(out)} exists already')
    return 'derive', f'{what} data of {os.path.basename(d)}', ['derive', what, d, '--out', out], {'memory_gb': 2.0}


@tool('jobs_derive', 'jobs', 'act',
      'Derive data from a drive: SYNTHETIC microphone-array audio with ground truth into the run (audio; the rig '
      'needs an audio_array), world-model data (occupancy, BEV, scene flow, futures, clips) or grounded language '
      'data (captions and QA from the simulator\'s truth) into derived/<run>-<what>.',
      C.obj({'what': C.s('what to derive', enum=['audio', 'worldmodel', 'language']), 'run': C.s('run id')},
            ['what', 'run']),
      preview=_cli_preview(_derive))
def jobs_derive(ctx, **a):
    return _queue(ctx, _derive(ctx, **a))


def _recon(ctx, run, iterations=7000, device='auto', seed=0, dry=False):
    from ...core.recon.train import job_spec
    d = C.resolve_run(ctx, run)
    out = os.path.join(C.workspace(ctx), 'recon', os.path.basename(d))
    if not dry and os.path.exists(os.path.join(out, 'recon.json')):
        raise ToolError(f'recon/{os.path.basename(d)} exists already')
    argv = ['train', '--source', f'falcon:{d}', '--out', out, '--iterations', str(iterations), '--device', device,
            '--seed', str(seed)]
    return 'recon', f'Reconstruct {os.path.basename(d)}', ['recon'] + argv, {
        'spec': None if dry else job_spec(argv, f'Reconstruct {os.path.basename(d)}')}


@tool('jobs_reconstruct', 'jobs', 'act',
      'Neural reconstruction (3D Gaussian splatting) of a recorded drive (Replay ▸ Reconstruction): a GPU job; '
      'output recon/<run>. Then jobs_hybrid composites CARLA\'s actors over it.',
      C.obj({'run': C.s('run id'), 'iterations': C.i('iterations (default 7000)', minimum=100, maximum=200000),
             'device': C.s('auto, cpu, cuda', enum=['auto', 'cpu', 'cuda']), 'seed': C.i('seed')}, ['run']),
      estimate=lambda ctx, **a: {'seconds': None, 'what': 'GPU time ∝ iterations'},
      preview=_cli_preview(_recon))
def jobs_reconstruct(ctx, **a):
    return _queue(ctx, _recon(ctx, **a))


def _hybrid(ctx, recon, run, method='depth', device='auto', dry=False):
    from ...core.drive import run_dir_name
    from ...core.recon.train import job_spec
    rec = C.resolve_in(ctx, 'recon', recon, 'reconstruction')
    d = C.resolve_run(ctx, run)
    if not os.path.isfile(os.path.join(d, 'frames.jsonl')) or not os.path.isdir(os.path.join(d, 'instance')):
        raise ToolError('the drive needs frames.jsonl and instance labels')
    out = os.path.join(C.workspace(ctx), 'runs', run_dir_name(f'hybrid-{os.path.basename(d)}'))
    argv = ['hybrid', rec, '--drive', d, '--out', out, '--method', method, '--device', device]
    return 'recon', f'Hybrid world of {os.path.basename(d)}', ['recon'] + argv, {
        'spec': None if dry else job_spec(argv, f'Hybrid world of {os.path.basename(d)}')}


@tool('jobs_hybrid', 'jobs', 'act',
      'A hybrid world: CARLA\'s actors of a recorded drive composited over a reconstruction, labels CARLA\'s; a GPU '
      'job writing a new run.',
      C.obj({'recon': C.s('reconstruction id (folder in recon/)'), 'run': C.s('a drive with instance labels'),
             'method': C.s('compositing method (default depth)'),
             'device': C.s('auto, cpu, cuda', enum=['auto', 'cpu', 'cuda'])}, ['recon', 'run']),
      preview=_cli_preview(_hybrid))
def jobs_hybrid(ctx, **a):
    return _queue(ctx, _hybrid(ctx, **a))


# ================================================================================================= stack, V2X, reports
def _suite(ctx, name, driver='follower', scenarios=('lead_brake', 'stalled_vehicle'), seeds=(0, 1, 2), world='',
           rig='', faults=None, mrm=False, degraded=False, min_ttc_s=1.0, dry=False):
    from ...core import faults as FL
    from ...core import stack as ST
    from ...core.provenance import write_json
    if C._bad_name(name) or '/' in name:
        raise ToolError(f'{name!r} is not a suite name')
    w, r = _world_rig(ctx, world, rig)
    doc = ST.new_suite(name, driver, list(scenarios), list(seeds), w['id'], r['id'])
    doc['criteria']['min_ttc_s'] = float(min_ttc_s) if min_ttc_s else None
    for f in faults or ():
        errs = FL.validate([f])
        if errs:
            raise ToolError('fault: ' + '; '.join(errs))
    for e in doc['entries']:
        if faults:
            e['faults'] = [dict(f) for f in faults]
        if mrm:
            e['mrm'] = True
    if degraded:
        if driver != 'follower':
            raise ToolError('degraded mode is the test stack\'s (driver follower)')
        doc['degraded_mode'] = True
    errs = ST.validate_suite(doc, C.project(ctx))
    if errs:
        raise ToolError('refused by the suite check: ' + '; '.join(errs[:6]))
    d = os.path.join(C.workspace(ctx), 'suites', name)
    spath = os.path.join(d, 'suite-request.json')
    path = C.project(ctx).path if dry else _saved(ctx)
    if not dry:
        os.makedirs(d, exist_ok=True)
        write_json(spath, doc)
    return 'stack-suite', f'Suite {name} ({driver})', ST.suite_run_argv(path or '<the project file>', spath, d,
                                                                        _fake(ctx)), {'memory_gb': 0.5}


@tool('jobs_stack_suite', 'jobs', 'act',
      'Run a stack test suite (Stack testing): scenarios × seeds driven by a stack (follower test stack, autopilot, '
      'or ros2 through the bridge), judged pass/fail (no collision, minimum TTC), with optional timed sensor faults '
      '[{sensor, kind, onset_s, …}], the minimal-risk-manoeuvre check and the test stack\'s degraded mode; Wilson '
      'intervals. Resumable; output suites/<name>; read with analysis_suite, then jobs_safety_case.',
      C.obj({'name': C.s('suite name'), 'driver': C.s('who drives', enum=['follower', 'autopilot', 'ros2']),
             'scenarios': C.arr(C.s('scenario key or id'), 'scenarios (default lead_brake, stalled_vehicle)'),
             'seeds': C.arr(C.i('seed'), 'seeds (default 0, 1, 2)'), 'world': C.s('world id'), 'rig': C.s('rig id'),
             'faults': C.arr({'type': 'object'}, 'timed sensor faults (core.faults)'),
             'mrm': C.b('check the minimal-risk manoeuvre'), 'degraded': C.b('the test stack\'s degraded mode'),
             'min_ttc_s': C.n('pass criterion: minimum TTC (0: none; default 1.0)', minimum=0)}, ['name']),
      estimate=lambda ctx, **a: _drive_estimate(ctx, _world_rig(ctx, a.get('world'), a.get('rig'))[1], 12 * 20,
                                                len(a.get('scenarios') or (1, 2)) * len(a.get('seeds') or (0, 1, 2))),
      preview=_cli_preview(_suite))
def jobs_stack_suite(ctx, **a):
    out = _queue(ctx, _suite(ctx, **a))
    out['suite'] = a['name']
    return out


def _safety_case(ctx, suite, studies=(), dry=False):
    from ...core import safetycase as SCS
    d = C.resolve_in(ctx, 'suites', suite, 'suite', ('results.jsonl',))
    odd = os.path.join(C.workspace(ctx), 'odd.json')
    if not os.path.isfile(odd):
        raise ToolError('no safety-case ODD yet: project_odd_set target safety_case writes one')
    errs = SCS.validate_odd(C.read_json(C.checked(ctx, odd), {}))
    if errs:
        raise ToolError('the ODD is not valid: ' + '; '.join(errs[:5]))
    argv = ['stack', 'report', d, '--odd', odd, '--out', os.path.join(d, 'safety-case')]
    for s in studies:
        argv += ['--study', C.resolve_in(ctx, 'studies', s, 'study', ('results.jsonl',))]
    return 'stack-report', f'Safety case of {suite}', argv, {'memory_gb': 0.5}


@tool('jobs_safety_case', 'jobs', 'act',
      'Build the SOTIF-style safety-case report (HTML + JSON) of a finished suite against the workspace\'s ODD '
      '(odd.json), with studies as evidence of triggering conditions. Output suites/<suite>/safety-case.',
      C.obj({'suite': C.s('suite id'), 'studies': C.arr(C.s('study id'), 'studies to include')}, ['suite']),
      preview=_cli_preview(_safety_case))
def jobs_safety_case(ctx, **a):
    return _queue(ctx, _safety_case(ctx, **a))


def _benefit(ctx, runs, dry=False):
    dirs = _runs(ctx, runs)
    for d in dirs:
        if not os.path.isfile(os.path.join(d, 'cpm.jsonl')):
            raise ToolError(f'{C.run_id(ctx, d)} recorded no roadside unit messages (cpm.jsonl)')
    out = os.path.join(C.workspace(ctx), 'v2x', f'benefit-{_stamp()}.json')
    if not dry:
        os.makedirs(os.path.dirname(out), exist_ok=True)
    return 'v2x', f'V2X benefit of {len(dirs)} run(s)', ['v2x', 'benefit'] + dirs + ['--out', out], {}


@tool('jobs_v2x_benefit', 'jobs', 'act',
      'Measure what roadside units add (V2X): the share of road users near the ego its own sensors saw against its '
      'sensors plus the units\' CPMs, and the time to first detection out of occlusion, pooled over runs that '
      'recorded CPMs. Output v2x/benefit-<stamp>.json (read it with analysis_v2x_benefit).',
      C.obj({'runs': C.arr(C.s('run id'), 'runs with cpm.jsonl', minItems=1, maxItems=200)}, ['runs']),
      preview=_cli_preview(_benefit))
def jobs_v2x_benefit(ctx, **a):
    return _queue(ctx, _benefit(ctx, **a))


def _report_source(ctx, kind, source):
    from ...core import reports as R
    srcs = R.sources(C.workspace(ctx)).get(kind) or []
    for label, path in srcs:
        if source in (label, os.path.basename(os.path.normpath(path)), os.path.relpath(path, C.workspace(ctx))):
            return C.checked(ctx, path)
    raise ToolError(f'no {kind} source {source!r}: ' + ', '.join(lb for lb, _ in srcs[:12]) or 'none')


def _report(ctx, kind, source, dry=False):
    from ...core import reports as R
    src = _report_source(ctx, kind, source)
    out = os.path.join(R.reports_dir(C.workspace(ctx)), R.default_name(kind, src))
    return 'report', f'Report: {kind} ({os.path.basename(src)})', ['report', kind, src, '--out', out], {
        'memory_gb': 0.5}


@tool('jobs_report', 'jobs', 'act',
      'Write a self-contained HTML report with provenance (Reports): of a loop iteration, an evaluation, a dataset, '
      'a study, a leaderboard, a rule check, a root-cause answer, a VLM evaluation, a deployment or a CI run. '
      'analysis_report_sources lists what can be reported on. Output reports/<stamp>-<kind>-<source>.html.',
      C.obj({'kind': C.s('report kind', enum=['iteration', 'evaluation', 'dataset', 'study', 'leaderboard', 'rules',
                                              'rootcause', 'vlm', 'deploy', 'ci']),
             'source': C.s('the source\'s label or folder name')}, ['kind', 'source']),
      preview=_cli_preview(_report))
def jobs_report(ctx, **a):
    return _queue(ctx, _report(ctx, **a))


def _why(ctx, run, failure='criticality', threshold=0.5, model='', dry=False):
    from ...core.rootcause import out_dir
    d = C.resolve_run(ctx, run)
    path = C.project(ctx).path if dry else _saved(ctx)
    label = os.path.basename(d)
    od = os.path.join(C.workspace(ctx), 'rootcause', f'<stamp>-{label}') if dry else out_dir(C.workspace(ctx), label)
    argv = ['why', 'case', path or '<the project file>', '--run', d, '--failure', failure, '--threshold',
            str(threshold), '--out', od]
    if failure != 'criticality':
        if not model:
            raise ToolError('a perception failure needs the model')
        argv += ['--model', _model(ctx, model)['id']]
    if _fake(ctx):
        argv += ['--fake-carla', _fake(ctx)]
    return 'why', f'Why did it fail? {label}', argv, {'memory_gb': 1.0}


@tool('jobs_why', 'jobs', 'act',
      '"Why did this fail?" for a scenario run (Rare cases ▸ Why): counterfactual re-runs removing one factor at a '
      'time (occlusion, range, lighting, fog, rain, appearance, sensor faults) — which removal fixes it and which '
      'alone fails. Output rootcause/<stamp>-<run>; read with analysis_rootcause.',
      C.obj({'run': C.s('a failing scenario run id'),
             'failure': C.s('criticality, or a perception failure kind (missed …)'),
             'threshold': C.n('criticality failure threshold (default 0.5)', minimum=0, maximum=1),
             'model': C.s('model id for a perception failure')}, ['run']),
      preview=_cli_preview(_why))
def jobs_why(ctx, **a):
    return _queue(ctx, _why(ctx, **a))


def _loop(ctx, action='next', until=1, dry=False):
    path = C.project(ctx).path if dry else _saved(ctx)
    args = {'next': ['next'], 'run': ['run', '--until', str(until)], 'new': ['new']}[action]
    argv = ['loop', path or '<the project file>'] + args + ['--no-save'] + (['--fake-carla', _fake(ctx)]
                                                                            if _fake(ctx) else [])
    return 'loop', f'Loop: {action}' + (f' to step {until}' if action == 'run' else ''), argv, {
        'memory_gb': 1.0, 'resumable': True}


def _loop_estimate(ctx, action='next', until=1):
    """The next step's cost as the Loop page computes it (core.loop.estimate): of the latest unfinished iteration's
    next step, else of picking a new one."""
    from ...core import loop as L
    svc = C.need_service(ctx)
    its = L.iterations(C.workspace(ctx))
    it = its[-1] if its else None
    if action != 'new' and it is not None and it.get('status') in ('running', 'failed', 'cancelled'):
        nxt = L.next_step(C.workspace(ctx), it)
        est = L.estimate(C.project(ctx), it, nxt, svc.throughput(),
                         budget_mpx=float(svc.settings.get('render_budget_mpx'))) if nxt else None
    else:
        est = L.estimate(C.project(ctx), None, 'pick', svc.throughput())
    if not est:
        return {'seconds': None, 'what': 'nothing left to run in this iteration'}
    return {'seconds': est.get('seconds'), 'what': C.clip(L.describe_estimate(est), 300)}


@tool('jobs_loop', 'jobs', 'act',
      'Advance the closed loop (Loop page): next (the next step: evaluate, find weak cells, search, generate, '
      'retrain, evaluate again), run until step N, or new (start an iteration). Needs the loop\'s model and ODD '
      '(project_odd_set). Resumable; one loop job at a time.',
      C.obj({'action': C.s('next, run or new', enum=['next', 'run', 'new']),
             'until': C.i('run: the step to stop after', minimum=1, maximum=20)}),
      estimate=_loop_estimate, preview=_cli_preview(_loop))
def jobs_loop(ctx, **a):
    from ...core.jobs import FINAL
    if any(j.spec.kind == 'loop' and j.state not in FINAL for j in C.need_queue(ctx).jobs):
        raise ToolError('a loop job is running already')
    return _queue(ctx, _loop(ctx, **a))


def _apply_edits(ctx, world, dry=False):
    """(world, import config, the world id it writes): its map edits applied into worlds/<id>/edit-<hash> (the map
    editor's Apply), or without edits a re-import with the current converter into a new world (Worlds ▸ Re-import)."""
    from ...core import mapedit as ME
    from ...core.project import new_id
    from ...core.worlds import world_dir
    _i, w = C.find(ctx, 'worlds', world, 'world')
    if w.get('kind') != 'osm':
        raise ToolError('only a world imported from OpenStreetMap can be re-converted')
    edits = (w.get('osm') or {}).get('edits') or []
    folder = C.checked(ctx, os.path.join(C.workspace(ctx), os.path.dirname(w['xodr'])))
    wid = w['id'] if edits else new_id('world')
    out = os.path.join(world_dir(C.workspace(ctx), wid), f'edit-{ME.edits_hash(edits)[:10]}') if edits else \
        world_dir(C.workspace(ctx), wid)
    try:
        return w, ME.import_config_for(w, folder, out, edits), wid
    except FileNotFoundError as e:
        raise ToolError(f'cannot re-convert: {e}') from e


def _apply_preview(ctx, world):
    w, cfg, wid = _apply_edits(ctx, world)
    if cfg['edits']:
        return f'apply {len(cfg["edits"])} map edit(s) to {w["name"]} (osm_import worker)'
    return f're-import {w["name"]} with the current converter into a new world {wid} (osm_import worker; the old ' \
           'world and its runs stay)'


@tool('jobs_apply_map_edits', 'jobs', 'act',
      'Re-convert an imported world with its map edits (the map editor\'s Apply), or, when it has none, re-import it '
      'with the current converter (Worlds ▸ Re-import: same area, centre, handedness and settings) into a new world: '
      'a job in CARLA\'s Python (no server). When done, project_world_add_imported with the returned world id takes '
      'the new conversion into the project.',
      C.obj({'world': C.s('imported world id')}, ['world']), preview=_apply_preview)
def jobs_apply_map_edits(ctx, world):
    w, cfg, wid = _apply_edits(ctx, world)
    title = f'Apply map edits: {w["name"]}' if cfg['edits'] else f'Re-import: {w["name"]}'
    job = C.need_service(ctx).submit_osm_import(cfg, title=title)
    ctx.emit('job_queued', {'job': job.id, 'title': job.spec.title})
    return {'job': job.id, 'title': job.spec.title, 'world': wid, 'reimport': not cfg['edits'],
            'next': 'jobs_wait, then project_world_add_imported with this world id' + (
                '' if cfg['edits'] else ' (added as a new world beside the old one)')}


# ================================================================================================= following jobs
@tool('jobs_list', 'jobs', 'read',
      'This session\'s jobs, newest first (id, kind, title, state, progress, step or why it waits), or with '
      'history true the workspace\'s finished jobs across restarts (kind, title, status, when, what its exit means, '
      'its last error).',
      C.obj({'active_only': C.b('only queued, waiting and running jobs'),
             'history': C.b('the workspace\'s job history instead'),
             'limit': C.i('rows (default 20)', minimum=1, maximum=50)}))
def jobs_list(ctx, active_only=False, history=False, limit=20):
    from ...core.jobs import FINAL
    if history:
        from ...core.jobhistory import JobHistory
        recs = JobHistory(C.workspace(ctx)).read(int(limit))
        return C.capped([{k: (C.clip(r.get(k), 200) if k == 'error' else r.get(k))
                          for k in ('kind', 'title', 'status', 'started', 'finished', 'elapsed_s', 'exit', 'error')
                          if r.get(k) is not None} for r in recs], int(limit))
    jobs = [j for j in reversed(C.need_queue(ctx).jobs) if not active_only or j.state not in FINAL]
    return C.capped([C.job_brief(ctx, j) for j in jobs], int(limit))


def _job(ctx, job):
    j = C.need_queue(ctx).get(int(job))
    if j is None:
        raise ToolError(f'no job {job} in this session (jobs_list)')
    return j


@tool('jobs_get', 'jobs', 'read',
      'One job: its state, progress, current step or why it waits, elapsed time and ETA, its metrics, the run it '
      'writes, its exit and last error, and the tail of its log.',
      C.obj({'job': C.i('job id'), 'log_lines': C.i('log lines (default 12, at most 60)', minimum=0, maximum=60)},
            ['job']))
def jobs_get(ctx, job, log_lines=C.LOG_LINES):
    return C.job_brief(ctx, _job(ctx, job), log=int(log_lines))


@tool('jobs_cancel', 'jobs', 'act',
      'Cancel a job: a queued one is dropped, a running worker is asked to stop and tidies the server first. '
      'Its partial output stays (drives and studies resume).',
      C.obj({'job': C.i('job id')}, ['job']),
      preview=lambda ctx, job: f'cancel job {job}: {_job(ctx, job).spec.title}')
def jobs_cancel(ctx, job):
    from ...core.jobs import FINAL
    j = _job(ctx, job)
    if j.state in FINAL:
        raise ToolError(f'job {job} has finished already ({j.state})')
    C.need_queue(ctx).cancel(j)
    return C.job_brief(ctx, j)


@tool('jobs_wait', 'jobs', 'read',
      'Wait until any of the given jobs is final (done, failed, cancelled) or timeout seconds pass (at most 600); '
      'returns each job\'s state and which finished. Waiting does not change anything. Call it again to keep '
      'waiting.',
      C.obj({'jobs': C.arr(C.i('job id'), 'job ids', minItems=1, maxItems=20),
             'timeout_s': C.n('seconds at most (default 60)', minimum=0, maximum=WAIT_MAX_S)}, ['jobs']))
def jobs_wait(ctx, jobs, timeout_s=60.0):
    """The queue's listeners set an Event when a watched job finishes; the caller's thread waits on it (never the GUI
    thread: the agent loop calls tools off it). From the command line nothing else drives the queue, so the waiting
    thread polls it between waits (as the CLI's own loop does)."""
    from ...core.jobs import FINAL
    q = C.need_queue(ctx)
    js = [_job(ctx, j) for j in jobs]
    if ctx.win is not None and threading.current_thread() is threading.main_thread() and timeout_s > 0:
        raise ToolError('jobs_wait blocks: call it off the window\'s thread (the agent loop does)')
    ev = threading.Event()
    ids = {j.id for j in js}

    def on(evname, job):
        if evname == 'finished' and job.id in ids:
            ev.set()
    q.subscribe(on)
    try:
        t0 = time.monotonic()
        while not any(j.state in FINAL for j in js):
            left = float(timeout_s) - (time.monotonic() - t0)
            if left <= 0:
                break
            if ctx.win is None:
                q.poll()
                ev.wait(min(0.25, left))
            else:
                ev.wait(min(1.0, left))
    finally:
        try:
            q.listeners.remove(on)
        except ValueError:
            pass
    done = [j.id for j in js if j.state in FINAL]
    return {'finished': done, 'timed_out': not done, 'jobs': [C.job_brief(ctx, j) for j in js]}
