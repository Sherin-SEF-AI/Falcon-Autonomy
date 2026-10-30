"""Lab tools: the longer workflows of the Hardware, Stack, Replay, Rare cases, Training and Models pages — a
hardware-in-the-loop session on the simulated device, a deployment to TensorRT engines (the simulated device, or the
Orin when the user configured it), CI mode headless, log replay (import, replay, compare), adversarial traffic (train,
evaluate), the mixed sim/real and active-learning experiments, and VLM explanation evaluation — and one reader of what
they wrote, which analysis_outputs lists and, with an id, reads (result).

Each acting tool builds what its page builds (core.hil.hil_config + submit_session; `falcon deploy all` as the Deploy
tab; a falcon.ci/1 config around a suite and `falcon ci run`; core.logreplay's make_plan, replay_config and submit as
the Replay page, `falcon replay import|compare`; `falcon adversary train|eval` as the Rare cases section; `falcon
experiment mixed|active` as the Training tabs; `falcon vlm eval` with core.vlm_eval.job_spec as the VLM tab) and
queues it on the app's queue: the same locks, memory guard, outputs and history. Nothing here needs the user's device
unless its address is already in their settings (Hardware page); no credential is ever asked for or shown. The HIL
session runs the device agent on this machine (the simulated device) only. VLM evaluations send images to the
runner (on this machine), never to the assistant: analysis_outputs returns their text scores only.
"""
from __future__ import annotations

import json
import os

from ..tool_api import ToolError, tool
from . import _common as C
from .analysis import OUTPUTS, compact
from .jobs import _cli_preview, _fake, _model, _new_dir, _queue, _real_source, _runs, _saved, _stamp, _world_rig


def _settings(ctx):
    return ctx.settings if ctx.win is None else ctx.win.settings


# ================================================================================================= hardware in the loop
def _hil_cfg(ctx, world='', rig='', seconds=30.0, camera='raw-tcp'):
    """The Hardware page's session config, device 'local' (the simulated device); raw-tcp needs no GStreamer."""
    from ...core.drive import run_dir_name
    from ...core.hil import device_settings, hil_config
    from ...core.project import ProjectError
    svc = C.need_service(ctx)
    w, r = _world_rig(ctx, world, rig)
    dev = device_settings(_settings(ctx)) if _settings(ctx) is not None else {'port': 7710}
    out = os.path.join(C.workspace(ctx), 'runs', run_dir_name('hil'))
    try:
        return hil_config(C.project(ctx), w['id'], r['id'], out, svc.port, svc.tm_port, seconds=float(seconds),
                          camera=camera, device='local', device_port=int(dev['port']), workspace=C.workspace(ctx))
    except (ProjectError, ValueError, KeyError) as e:
        raise ToolError(str(e)) from e


@tool('lab_hil', 'lab', 'act',
      'A hardware-in-the-loop session on the SIMULATED device (Hardware ▸ Start, device "this machine"): the rig\'s '
      'first camera (raw frames over TCP) and lidar streamed to the device agent run here, its controls '
      'back; per-stage latency, the deadline counts and the clock-offset estimate in run.json and report.html. A real '
      'device is started from the Hardware page by the user. Read with analysis_outputs kind hil and the run id.',
      C.obj({'world': C.s('world id'), 'rig': C.s('rig id (needs a pinhole camera)'),
             'seconds': C.n('session length (default 30)', minimum=1, maximum=600)}),
      estimate=lambda ctx, **a: {'seconds': None, 'what': f'{float(a.get("seconds", 30)):g} s of session at the '
                                                          'simulation\'s real-time pace, plus starting the server and '
                                                          'the device agent'},
      preview=lambda ctx, **a: (lambda c: f'HIL session on the simulated device: {c["seconds"]:g} s at '
                                          f'{c["fps"]:g} Hz, camera {c["camera"]["sensor"]["width"]}×'
                                          f'{c["camera"]["sensor"]["height"]} over {c["camera"]["transport"]}, '
                                          f'lidar {c["lidar"]["sensor"].get("channels")} channels → runs/'
                                          f'{os.path.basename(c["out"])}')(_hil_cfg(ctx, **a)))
def lab_hil(ctx, **a):
    from ...core.hil import submit_session
    cfg = _hil_cfg(ctx, **a)
    try:
        job = submit_session(C.need_service(ctx), cfg, '')
    except (RuntimeError, ValueError) as e:
        raise ToolError(str(e)) from e
    ctx.emit('job_queued', {'job': job.id, 'title': job.spec.title})
    return {'job': job.id, 'title': job.spec.title, 'run': C.run_id(ctx, cfg['out']),
            'follow': 'jobs_wait, then analysis_outputs kind hil with the run id'}


# ================================================================================================= deployment
def _deploy(ctx, model, runs, calib_runs=(), precisions=('fp32', 'fp16'), device='sim', dry=False):
    from ...core import deploy as DP
    from ...core.hil import device_settings
    ev, cal = _runs(ctx, runs), [C.resolve_run(ctx, r) for r in calib_runs]      # the paths first (privacy)
    m = _model(ctx, model)
    if m.get('kind') != 'detector' or m.get('runtime') != 'onnx':
        raise ToolError(f'{m["id"]} is not an ONNX detector (the Deploy tab deploys those)')
    if device == 'orin' and not (device_settings(_settings(ctx)) if _settings(ctx) is not None else {}).get('address'):
        raise ToolError('the Orin is not configured (its address is set by the user on the Hardware page); use the '
                        'simulated device')
    if 'int8' in precisions and not calib_runs:
        raise ToolError('INT8 needs calibration drives (other drives than the evaluation\'s)')
    path = C.project(ctx).path if dry else _saved(ctx)
    name = f'{_stamp()}-{m["id"]}'
    out = os.path.join(DP.deploy_dir(C.workspace(ctx)), name) if dry else _new_dir(ctx, 'deploy', name)
    argv = ['deploy', 'all', path or '<the project file>', '--model', m['id'], '--out', out, '--device', device,
            '--precision', ','.join(precisions)]
    for r in ev:
        argv += ['--run', r]
    for r in cal:
        argv += ['--calib-run', r]
    where = 'the simulated device' if device == 'sim' else 'the Jetson'
    return 'deploy', f'Deploy {m.get("name")} on {where}', argv, {'memory_gb': 2.0}


@tool('lab_deploy', 'lab', 'act',
      'Deploy a registered ONNX detector as TensorRT engines and gate it (Hardware ▸ Deploy, `falcon deploy all`): '
      'build per precision, profile (latency, throughput, power), accuracy against the host reference with paired '
      'intervals, the default release gate and the report. On the simulated device (onnxruntime on this CPU, '
      'labelled: no Orin number) unless the user configured the Orin. Output deploy/<id>; read with analysis_outputs.',
      C.obj({'model': C.s('model id (an ONNX detector)'), 'runs': C.arr(C.s('run id'), 'evaluation drives', minItems=1,
                                                                    maxItems=100),
             'calib_runs': C.arr(C.s('run id'), 'INT8 calibration (int8 needs them)', maxItems=100),
             'precisions': C.arr(C.s('precision', enum=['fp32', 'fp16', 'int8']), 'engines (default fp32, fp16)',
                                 minItems=1),
             'device': C.s('sim (default) or orin (only when configured)', enum=['sim', 'orin'])},
            ['model', 'runs']),
      estimate=lambda ctx, **a: {'seconds': None, 'what': 'build and 1000 timed runs per engine, then inference '
                                                          'over the evaluation frames'},
      preview=_cli_preview(_deploy))
def lab_deploy(ctx, **a):
    out = _queue(ctx, _deploy(ctx, **a))
    out['deployment'] = os.path.basename(out['command'].split('--out ', 1)[1].split(' ', 1)[0])
    return out


# ================================================================================================= CI mode
def _ci(ctx, suite, name, min_pass_rate=0.9, rule='refute', baseline='', dry=False):
    from ...core.provenance import write_json
    sp = os.path.join(C.resolve_in(ctx, 'suites', suite, 'suite', ('suite-request.json',)), 'suite-request.json')
    out = os.path.join(C.workspace(ctx), 'ci', name) if dry else C.out_dir(ctx, 'ci', name)
    cfg = {'schema': 'falcon.ci/1', 'name': name, 'project': C.project(ctx).path if dry else _saved(ctx),
           'subject': {'kind': 'stack', 'suite': sp}, 'carla': {'mode': 'fake' if _fake(ctx) else 'falcon'},
           'gates': {'pass_rate': {'min': float(min_pass_rate), 'rule': rule}}}
    if baseline:
        cfg['baseline'] = {'kind': 'dir', 'dir': C.resolve_in(ctx, 'ci', baseline, 'CI run', ('ci-results.json',))}
        cfg['gates']['regressions'] = {}
    cpath = os.path.join(C.workspace(ctx), 'ci-configs', f'{name}.json')
    if not dry:
        os.makedirs(os.path.dirname(cpath), exist_ok=True)
        write_json(cpath, cfg)
    argv = ['ci', 'run', cpath, '--out', out, '--quiet'] + (['--fake-carla', _fake(ctx)] if _fake(ctx) else [])
    return 'ci', f'CI: {name}', argv, {'memory_gb': 0.5}


def _ci_estimate(ctx, suite, **_a):
    """The suite's rendering cost on this machine (core.budget through the service's throughput): every entry's
    seeds × its scenario's steps."""
    from .jobs import _drive_estimate, _scenario
    d = C.resolve_in(ctx, 'suites', suite, 'suite', ('suite-request.json',))
    doc = C.read_json(os.path.join(d, 'suite-request.json'), {}) or {}
    _w, r = _world_rig(ctx, doc.get('world', ''), doc.get('rig', ''))
    fps = float(doc.get('fps') or 20.0)
    steps, n = 0, 0
    for e in doc.get('entries') or []:
        k = len(e.get('seeds') or [0])
        try:
            dur = float(_scenario(ctx, e['scenario']).get('duration_s', 12))
        except (ToolError, KeyError):
            dur = 12.0
        steps, n = steps + int(dur * fps) * k, n + k
    if not n:
        return {'seconds': None, 'what': 'the suite has no entries'}
    est = _drive_estimate(ctx, r, max(1, steps // n), n)
    return dict(est, what=f'{n} scenario run(s) of the suite: ' + est['what'])


@tool('lab_ci', 'lab', 'act',
      'CI mode headless (`falcon ci run`): a finished or planned stack suite (suites/<id>, from jobs_stack_suite) run '
      'as a CI job with a pass-rate gate judged on its Wilson interval (demonstrate, refute or estimate) and, against '
      'a previous CI run, a no-regression gate; writes ci-results.json, JUnit, the Markdown summary and the report '
      'into ci/<name>. Never posts anywhere. Read with analysis_outputs kind ci.',
      C.obj({'suite': C.s('suite id'), 'name': C.s('its name (folder in ci/)'),
             'min_pass_rate': C.n('pass-rate gate (default 0.9)', minimum=0, maximum=1),
             'rule': C.s('the interval rule (default refute: fails only on a shortfall shown at 95 %)',
                         enum=['demonstrate', 'refute', 'estimate']),
             'baseline': C.s('a previous CI run id')}, ['suite', 'name']),
      estimate=_ci_estimate, preview=_cli_preview(_ci))
def lab_ci(ctx, **a):
    out = _queue(ctx, _ci(ctx, **a))
    out['ci'] = a['name']
    return out


# ================================================================================================= log replay
def _log_path(ctx, log):
    return os.path.join(C.resolve_in(ctx, 'logs', log, 'log', ('log.json',)), 'log.json')


def _replay(ctx, action, source='', log='', runs=(), world='', rig='', driver='log', dry=False):
    import re
    runs = [C.resolve_run(ctx, r) for r in runs]
    if action == 'import':
        from ...core.logreplay import KINDS
        if ':' not in source:
            raise ToolError('source is KIND:PATH inside the workspace (' + ', '.join(KINDS) + ')')
        kind, rel = source.split(':', 1)
        if kind not in KINDS:
            raise ToolError(f'kind must be one of {", ".join(KINDS)}')
        if C._bad_name(rel):
            raise ToolError(f'{rel!r} is not a path inside the workspace')
        p = C.checked(ctx, os.path.join(C.workspace(ctx), rel))
        if not os.path.exists(p):
            raise ToolError(f'no {rel} in the workspace')
        nm = re.sub(r'[^A-Za-z0-9_.-]+', '-', os.path.splitext(os.path.basename(p))[0]).strip('-').lower()[:60]
        out = C.out_dir(ctx, 'logs', nm or 'log')
        return 'replay', f'Import log {nm}', ['replay', 'import', f'{kind}:{p}', '--out', out], {'memory_gb': 1.0}
    if action == 'compare':
        if len(runs) != 2:
            raise ToolError('compare takes runs [original, changed]: two replays')
        a, b = runs
        out = b.rstrip('/') + '-comparison'
        if not dry and os.path.exists(out):
            raise ToolError(f'{C.run_id(ctx, out)} exists already')
        return 'replay', f'Compare replays {os.path.basename(b)}', ['replay', 'compare', '--original', a,
                                                                    '--changed', b, '--out', out], {'memory_gb': 1.0}
    raise ToolError('action is import, run or compare')


def _replay_cfg(ctx, log, world='', rig='', driver='log'):
    from ...core.drive import run_dir_name
    from ...core.logreplay import Log, LogError, make_plan, world_info
    from ...core.logreplay.commands import replay_config
    svc = C.need_service(ctx)
    w, r = _world_rig(ctx, world, rig)
    lp = _log_path(ctx, log)
    try:
        lg = Log.load(lp)
        plan = make_plan(lg, world_info(C.project(ctx), w['id'], C.workspace(ctx)), driver=driver, log_path=lp)
        out = os.path.join(C.workspace(ctx), 'runs', run_dir_name('replay-original'))
        cfg = replay_config(C.project(ctx), plan, w['id'], r['id'], out, svc.port, svc.tm_port,
                            float(svc.settings.get('render_budget_mpx')), C.workspace(ctx))
    except (LogError, ValueError, KeyError) as e:
        raise ToolError(f'cannot replay: {e}') from e
    return cfg, lg, plan, r


def _replay_preview(ctx, action, **a):
    if action != 'run':
        return C.command_line(_replay(ctx, action, dry=True, **a)[2])
    cfg, lg, plan, _r = _replay_cfg(ctx, a.get('log', ''), a.get('world', ''), a.get('rig', ''),
                                    a.get('driver', 'log'))
    return (f'replay {lg.meta.get("name")} in {cfg["name"]}: {cfg["steps"]} steps at {cfg["fps"]:g} Hz, placement '
            f'{plan["placement"]["mode"]}, driver {plan["ego"]["driver"]} → runs/{os.path.basename(cfg["out"])}')


def _replay_estimate(ctx, action, **a):
    if action != 'run':
        return {'seconds': None, 'what': 'reading the log (import) or the runs\' frames (compare)'}
    from .jobs import _drive_estimate
    cfg, _lg, _p, r = _replay_cfg(ctx, a.get('log', ''), a.get('world', ''), a.get('rig', ''), a.get('driver', 'log'))
    return _drive_estimate(ctx, r, cfg['steps'])


@tool('lab_replay', 'lab', 'act',
      'Log replay (Replay page): import a log (nuscenes, waymo, ros2, kitti_raw or falcon:PATH inside the workspace) '
      'into logs/<name>; run it in CARLA as a CARLA job (a world and rig, the log\'s ego driving, or the follower) '
      'into a new run; or compare two replays (original, changed) by driving behaviour into <changed>-comparison.',
      C.obj({'action': C.s('what to do', enum=['import', 'run', 'compare']),
             'source': C.s('import: KIND:PATH in the workspace'), 'log': C.s('run: log id (folder in logs/)'),
             'runs': C.arr(C.s('run id'), 'compare: [original, changed]'), 'world': C.s('world id'),
             'rig': C.s('rig id'), 'driver': C.s('run: who drives (default the log)', enum=['log', 'follower'])},
            ['action']),
      estimate=_replay_estimate, preview=_replay_preview)
def lab_replay(ctx, action, **a):
    for r in a.get('runs') or ():                    # every run id checked (privacy) before anything else
        C.resolve_run(ctx, r)
    if action != 'run':
        out = _queue(ctx, _replay(ctx, action, **a))
        if action == 'import':
            out['log'] = os.path.basename(out['command'].split('--out ', 1)[1].split(' ', 1)[0])
        return out
    from ...core.logreplay.commands import submit
    if not a.get('log'):
        raise ToolError('run needs a log id (analysis_outputs kind log lists them)')
    cfg, lg, _p, _r = _replay_cfg(ctx, a['log'], a.get('world', ''), a.get('rig', ''), a.get('driver', 'log'))
    try:
        job = submit(C.need_service(ctx), cfg, f'Replay (original): {lg.meta.get("name")}')
    except RuntimeError as e:
        raise ToolError(str(e)) from e
    ctx.emit('job_queued', {'job': job.id, 'title': job.spec.title})
    return {'job': job.id, 'title': job.spec.title, 'run': C.run_id(ctx, cfg['out'])}


# ================================================================================================= adversarial traffic
def _ego(ctx, ego):
    if ego in ('follower', 'none', 'autopilot'):
        return ego
    return C.resolve_in(ctx, 'policies', ego, 'policy')


def _adversary(ctx, action, ego='follower', updates=50, episodes=20, policy='', dry=False):
    path = C.project(ctx).path if dry else _saved(ctx)
    e = _ego(ctx, ego)
    root = os.path.join(C.workspace(ctx), 'adversary')
    if action == 'train':
        out = os.path.join(root, f'train-{_stamp()}')
        argv = ['adversary', 'train', path or '<the project file>', '--out', out, '--ego', e, '--action', 'lateral',
                '--updates', str(int(updates))]
        title = f'Train adversaries against {ego}'
    else:
        if not policy:
            raise ToolError('eval needs policy: a training id (analysis_outputs kind adversary lists them)')
        pol = os.path.join(C.resolve_in(ctx, 'adversary', policy, 'adversary training'), 'policy')
        if not os.path.isfile(os.path.join(pol, 'policy.json')):
            raise ToolError(f'adversary/{policy} has no trained policy yet')
        out = os.path.join(root, f'eval-{_stamp()}')
        argv = ['adversary', 'eval', path or '<the project file>', '--policy', pol, '--out', out, '--ego', e,
                '--episodes', str(int(episodes))]
        title = f'Evaluate adversaries ({policy}) against {ego}'
    if _fake(ctx):
        argv += ['--fake-carla', _fake(ctx)]
    return 'adversary', title, argv, {'memory_gb': 1.0}


@tool('lab_adversary', 'lab', 'act',
      'Learned adversarial traffic (Rare cases ▸ Adversarial traffic): train road users with multi-agent PPO to make '
      'a fixed ego fail realistically (train), or evaluate trained ones against random, idle and scripted searches '
      'with the same budget: failure rates with Wilson intervals, realism metrics, replayable cases (eval). Output '
      'adversary/train-… or eval-…; read with analysis_outputs kind adversary.',
      C.obj({'action': C.s('train or eval', enum=['train', 'eval']),
             'ego': C.s('follower, none, autopilot or a policy id (default follower)'),
             'updates': C.i('train: PPO updates (default 50)', minimum=1, maximum=100000),
             'episodes': C.i('eval: per method (default 20)', minimum=1, maximum=10000),
             'policy': C.s('eval: training id')}, ['action']),
      estimate=lambda ctx, **a: {'seconds': None, 'what': 'CARLA steps: updates × 512 (train) or episodes × 5 '
                                                          'methods (eval)'},
      preview=_cli_preview(_adversary))
def lab_adversary(ctx, **a):
    return _queue(ctx, _adversary(ctx, **a))


# ================================================================================================= experiments
def _experiment(ctx, kind, name, runs, test_runs=(), real_train='', real_test='', seeds=3, epochs=6, dry=False):
    from ...core import training as T
    if C._bad_name(name) or '/' in name:
        raise ToolError(f'{name!r} is not an experiment name')
    dirs = _runs(ctx, runs)
    argv = ['experiment', kind, '--workspace', C.workspace(ctx), '--id', name]
    if kind == 'mixed':
        if not real_train or not real_test:
            raise ToolError('mixed needs real_train and real_test (KIND:FOLDER inside the workspace)')
        argv += ['--real-train', _real_source(ctx, real_train), '--real-test', _real_source(ctx, real_test),
                 '--shares', '0,25,50,100', '--seeds', str(int(seeds)), '--epochs', str(int(epochs))]
        argv += [x for r in dirs for x in ('--synthetic', f'falcon:{r}')]
        title = f'Mixed experiment {name}'
    else:
        if not test_runs:
            raise ToolError('active needs test_runs: held-out drives, none of them in the pool')
        tests = _runs(ctx, test_runs)
        if set(tests) & set(dirs):
            raise ToolError('a test drive cannot also be in the pool')
        argv += ['--strategies', 'random,entropy,kcenter,hybrid', '--initial', '32', '--budget', '32', '--rounds',
                 '3', '--seeds', str(int(seeds)), '--epochs', str(int(epochs)), '--arch', next(iter(T.ARCHS)),
                 '--init', 'coco']
        argv += [x for r in dirs for x in ('--pool', f'falcon:{r}')]
        argv += [x for r in tests for x in ('--test', f'falcon:{r}')]
        title = f'Active learning {name}'
    return 'experiment', title, argv, {'needs_gpu': True, 'memory_gb': 6.0, 'env': dict(os.environ,
                                                                                         FALCON_GPU_LOCK_HELD='1')}


@tool('lab_experiment', 'lab', 'act',
      'A training experiment (Training ▸ Experiments, Active learning): mixed — real only against real + 25/50/100 % '
      'synthetic drives at a fixed budget, K seeds, scored on a held-out real test set with honest intervals; '
      'active — frames chosen by uncertainty or diversity against random choice, over rounds and seeds, on held-out '
      'drives. GPU jobs, resumable; pretrained weights must already be fetched. Read with analysis_outputs.',
      C.obj({'kind': C.s('mixed or active', enum=['mixed', 'active']), 'name': C.s('its id (folder in experiments/)'),
             'runs': C.arr(C.s('run id'), 'synthetic drives, or the active pool', minItems=1, maxItems=500),
             'test_runs': C.arr(C.s('run id'), 'active: held-out drives', maxItems=500),
             'real_train': C.s('mixed: KIND:FOLDER (in the workspace)'),
             'real_test': C.s('mixed: KIND:FOLDER, held out'),
             'seeds': C.i('seeds (default 3)', minimum=1, maximum=20),
             'epochs': C.i('epochs (default 6)', minimum=1, maximum=200)}, ['kind', 'name', 'runs']),
      estimate=lambda ctx, **a: {'seconds': None, 'what': 'GPU training: arms (or strategies × rounds) × seeds × '
                                                          'epochs; the job reports its ETA once running'},
      preview=_cli_preview(_experiment))
def lab_experiment(ctx, **a):
    out = _queue(ctx, _experiment(ctx, **a))
    out['experiment'] = a['name']
    return out


# ================================================================================================= VLM evaluation
def _vlm(ctx, runs, runner='oracle', every=5, max_frames=0, dry=False):
    from ...core import vlm_eval as E
    from ...core import vlm_models as VM
    if runner.startswith('hf:'):
        name = runner[3:]
        if name not in VM.CANDIDATES:
            raise ToolError(f'hf runners are the listed candidates ({", ".join(VM.CANDIDATES) or "none"})')
        if not os.path.isfile(os.path.join(VM.candidate_dir(C.workspace(ctx), name), VM.PINS)):
            raise ToolError(f'{name} is not downloaded: the user fetches it on Models ▸ VLM (its licence is theirs '
                            'to accept)')
    elif runner not in ('oracle', 'noisy'):
        raise ToolError('runner is oracle, noisy or hf:<downloaded candidate>')
    dirs = _runs(ctx, runs)
    nm = runner.replace(':', '-')[:32]
    base = os.path.join(E.evaluations_dir(C.workspace(ctx)), f'{_stamp()}-{nm}')
    out = base if dry else _new_dir(ctx, os.path.join('vlm', 'evaluations'), f'{_stamp()}-{nm}')
    argv = ['eval', '--runner', runner, '--out', out, '--workspace', C.workspace(ctx), '--every', str(int(every)),
            '--max-frames', str(int(max_frames))]
    for r in dirs:
        argv += ['--run', r]
    gpu = E.needs_gpu(runner)
    title = f'VLM evaluation: {nm} on {len(dirs)} drive(s)'
    return 'vlm', title, ['vlm'] + argv, {'spec': None if dry else E.job_spec(argv, title, needs_gpu=gpu,
                                                                            memory_gb=6.0 if gpu else 1.5)}


@tool('lab_vlm', 'lab', 'act',
      'Evaluate a vision-language model\'s driving explanations against the simulator\'s truth (Models ▸ VLM, `falcon '
      'vlm eval`): grounded questions per frame of drives, accuracy per type with intervals over drives, grounding, '
      'hallucination rate. Runners: oracle and noisy (test baselines, not models) or a downloaded candidate. The '
      'images go to the runner on this machine only. Output vlm/evaluations/<id>; read with analysis_outputs.',
      C.obj({'runs': C.arr(C.s('run id'), 'drives', minItems=1, maxItems=100),
             'runner': C.s('oracle, noisy or hf:<candidate> (default oracle)'),
             'every': C.i('every Nth frame (default 5)', minimum=1, maximum=1000),
             'max_frames': C.i('per drive (default 0: all)', minimum=0, maximum=100000)}, ['runs']),
      estimate=lambda ctx, **a: {'seconds': None, 'what': 'a model per question; the baselines take seconds'},
      preview=_cli_preview(_vlm))
def lab_vlm(ctx, **a):
    out = _queue(ctx, _vlm(ctx, **a))
    out['evaluation'] = os.path.basename(out['command'].split('--out ', 1)[1].split(' ', 1)[0])
    return out


# ================================================================================================= results
def _hil(doc):
    h = doc.get('hil') or {}
    if not h:
        raise ToolError('the run holds no HIL session summary (still running, or not a HIL session)')
    keep = ('label', 'median', 'p95', 'p99', 'clock', 'trusted')
    return {'stages_ms': {k: {x: st.get(x) for x in keep if x in st} for k, st in (h.get('stages') or {}).items()},
            'measures_ms': {k: {x: m.get(x) for x in ('label', 'n', 'median', 'p95', 'p99')}
                            for k, m in (h.get('measures') or {}).items()},
            'deadline': h.get('deadline'), 'frames': h.get('frames'), 'drops': h.get('drops'),
            'clock_offset': h.get('clock_offset'), 'one_way_trusted': h.get('one_way_trusted'),
            'device': (h.get('session') or {}).get('device'), 'notes': h.get('notes')}


def result(ctx, kind, id):
    """One lab output's numbers (analysis_outputs with an id): a CI run's verdict and gates, a HIL session's latency
    per stage and deadlines, a deployment's engines, an adversary evaluation's failure rates (or a training's
    metrics), an experiment's arms or strategies, a VLM evaluation's scores (text only), a log's actors, a replay
    comparison's differences: the files' own numbers, compacted."""
    folder, must = OUTPUTS[kind]
    if kind == 'hil':
        d = C.resolve_run(ctx, id)
    else:
        d = C.resolve_in(ctx, folder, os.path.basename(id) if kind == 'replay_comparison' else id,
                         kind.replace('_', ' '), must)
    rd = lambda f: C.read_json(os.path.join(d, f)) or {}     # noqa: E731
    if kind == 'hil':
        r = rd('run.json')
        return C.rounded(dict(_hil(r), run=C.run_id(ctx, d), status=r.get('status'), error=r.get('error')), 3)
    if kind == 'ci':
        r = rd('ci-results.json')
        return compact({'name': r.get('name'), 'verdict': r.get('verdict'), 'error': r.get('error'),
                        'subject': r.get('subject'), 'deterministic': r.get('deterministic'),
                        'runs': len(r.get('runs') or []), 'summary': (r.get('summary') or {}).get('overall'),
                        'gates': [{k: g.get(k) for k in ('name', 'status', 'measured', 'limit', 'rule', 'detail')
                                   if g.get(k) not in (None, '')} for g in r.get('gates') or []],
                        'comparison': {k: (r.get('comparison') or {}).get(k) for k in ('pairs', 'regressions')}
                        if r.get('comparison') else None}, cap=20)
    if kind == 'deploy':
        from ...core import deploy as DP
        doc = rd('deploy.json')
        return compact({'simulated': doc.get('simulated'), 'label': doc.get('label'),
                        'model': (doc.get('model') or {}).get('id'),
                        'device': {k: (doc.get('device') or {}).get(k) for k in ('kind', 'model', 'backend')},
                        'engines': [{k: v for k, v in e.items() if v not in (None, [None, None])}
                                    for e in DP.engine_rows(doc)] if doc.get('engines') else [],
                        'stages': list((doc.get('stages') or {}).keys()), 'notes': doc.get('notes')}, cap=12)
    if kind == 'adversary':
        ev = rd('adversary-eval.json')
        if ev:
            short = {ego: {m: {k: x.get(k) for k in ('episodes', 'realistic_failures', 'failure_rate',
                                                    'failure_rate_ci95', 'distinct_failures')}
                           for m, x in res.items()} for ego, res in (ev.get('results') or {}).items()}
            return compact({'evaluation': id, 'results': short, 'failures': ev.get('failures'),
                            'realistic_failures': ev.get('realistic_failures'), 'failed': ev.get('failed'),
                            'method_note': ev.get('method_note')}, cap=12)
        pol = C.read_json(os.path.join(d, 'policy', 'metrics.json')) or {}
        return compact({'training': id, 'policy': os.path.isfile(os.path.join(d, 'policy', 'policy.json')),
                        'metrics': pol}, cap=12)
    if kind == 'log':
        from ...core.logreplay import Log, LogError
        try:
            lg = Log.load(os.path.join(d, 'log.json'))
        except (LogError, OSError, ValueError) as e:
            raise ToolError(f'not a readable log: {e}') from e
        return C.rounded({'log': id, 'name': lg.meta.get('name'), 'source_kind': lg.meta.get('kind'),
                          'map': {k: lg.map.get(k) for k in ('kind', 'name')}, **lg.counts()}, 3)
    doc = rd({'experiment': 'results.json', 'vlm': 'scores.json', 'replay_comparison': 'comparison.json'}[kind])
    if not doc:
        n = len([x for x in os.listdir(os.path.join(d, 'runs'))]) if os.path.isdir(os.path.join(d, 'runs')) else 0
        raise ToolError(f'no results yet ({n} run(s) so far; the analysis is written when the job finishes)')
    doc = {k: v for k, v in doc.items() if k not in ('provenance', 'schema')}
    if kind == 'vlm':
        doc.pop('rows', None)
    return compact(json.loads(json.dumps(doc, default=str)), cap=10)
