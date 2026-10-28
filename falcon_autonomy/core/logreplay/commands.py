"""`falcon replay import|match|run|compare` and the replay job (standard library; numpy through the package).

    falcon replay import [KIND:]PATH --out DIR [--scene S] [--segment S] [--town T] [--world-frame F] [--ego-frame F]
    falcon replay match LOG PROJECT [--world W]
    falcon replay run PROJECT --log LOG --out DIR [--world W] [--rig R] [--changes FILE] [--seed N] [--fps F]
                      [--driver log|follower|plugin] [--policy MODULE:CLASS] [--t-start S] [--t-end S]
                      [--anchor-spawn K] [--ego-offset M] [--allow-mismatch] [--keep-server]
    falcon replay compare --original RUN --changed RUN [RUN...] --out DIR [--project P --model ID |
                          --synthetic-detector] [--camera C] [--every N] [--boot N] [--seed S]

A replay runs as a CARLA job like a drive: replay_config() builds a drive config (the world's weather — with a
weather change applied — the rig, the plan's step, no background traffic, the plan) and submit() writes it into the
run folder as drive.json and queues the 'replay' worker through CarlaService (the CARLA lock, the memory guard,
Falcon's own server).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from .. import protocol as P
from .model import Log, LogError


def replay_config(project, plan: dict, world_id: str, rig_id: str, out_dir: str, port: int, tm_port: int,
                  render_budget_mpx: float = 200.0, workspace: str = '', name: str = '') -> dict:
    """A drive config running the plan (workers.replay)."""
    from .. import weather as W
    from ..drive import drive_config
    steps_s = plan['steps'] / plan['fps']
    cfg = drive_config(project, world_id, rig_id, out_dir, port, tm_port, seconds=steps_s, fps=plan['fps'],
                       seed=int(plan.get('seed', 0)), render_budget_mpx=render_budget_mpx, workspace=workspace,
                       name=name or f'replay: {plan["log"].get("name")}' + (' (changed)' if plan['changes'] else ''))
    cfg['steps'] = int(plan['steps'])
    cfg['traffic'] = dict(cfg['traffic'], vehicles=0, walkers=0)
    cfg['warmup_steps'] = 5
    wc = plan.get('weather')
    if wc:
        spec = dict(cfg.get('weather_source') or {})
        if wc.get('preset'):
            spec = {'preset': wc['preset']}
        spec.update(wc.get('params') or {})
        try:
            cfg['weather'] = W.resolve(spec)
        except (KeyError, ValueError) as e:
            raise LogError(f'weather change: {e}') from None
        cfg['weather_source'] = spec
    cfg['replay'] = {'plan': plan}
    return cfg


def submit(svc, cfg: dict, title: str = ''):
    """Write cfg into its run folder (drive.json) and queue the replay worker."""
    from ..paths import ensure_dir
    from ..provenance import write_json
    run_dir = ensure_dir(cfg['out'])
    path = os.path.join(run_dir, 'drive.json')
    write_json(path, cfg)
    spec = svc.worker_spec('replay', [path], title or cfg.get('name', 'replay'), 'replay', run_dir,
                           meta={'config': path})
    return svc.queue.submit(spec)


def lanes_for(world: dict, workspace: str = ''):
    """(lanes (N, 4), source) for an offline match: the world's OpenDRIVE file, or the lanes.json of a run of the same
    town in the workspace; (None, why) when neither exists (the replay worker measures it on the server)."""
    from . import mapmatch as MM
    from .plan import town_key
    if world.get('xodr') and os.path.isfile(world['xodr']):
        with open(world['xodr'], encoding='utf-8') as fh:
            return MM.lanes_from_xodr(fh.read()), f'the world\'s OpenDRIVE file ({os.path.basename(world["xodr"])})'
    if world.get('town') and workspace:
        runs = os.path.join(workspace, 'runs')
        for d in sorted(os.listdir(runs)) if os.path.isdir(runs) else []:
            lj, dj = os.path.join(runs, d, 'lanes.json'), os.path.join(runs, d, 'drive.json')
            if os.path.isfile(lj) and os.path.isfile(dj):
                with open(dj, encoding='utf-8') as fh:
                    town = json.load(fh).get('town')
                if town_key(town) == town_key(world['town']):
                    with open(lj, encoding='utf-8') as fh:
                        return MM.lanes_from_lanes_json(json.load(fh)), f'lanes.json of the run {d}'
    return None, 'no lanes known offline for this world (no OpenDRIVE file, no labelled run of its town): the ' \
                 'replay measures the match on the server'


def offline_match(log: Log, world: dict, workspace: str = '', anchor: dict | None = None) -> dict:
    """Placement and, when lanes are known offline, the measured match of the (placed) ego path."""
    from . import mapmatch as MM
    from .plan import geo_transform, resolve_placement
    pl = resolve_placement(log, world, anchor)
    out = {'placement': pl, 'match': None}
    if pl['mode'] == 'anchored':
        out['match'] = {'verdict': 'none', 'why': pl['why']}
        return out
    placed = geo_transform(log, world['geo']) if pl['mode'] == 'geo' else log
    lanes, src = lanes_for(world, workspace)
    if lanes is None:
        out['match_note'] = src
        return out
    out['match'] = MM.match(placed.ego.x, placed.ego.y, placed.ego.yaw, lanes, src)
    return out


def _wait(svc, job, em) -> int:
    from ..jobs import FINAL
    import time
    shown = 0
    try:
        while job.state not in FINAL:
            svc.queue.poll()
            if job.progress is not None:
                em.progress(job.progress, job.summary())
            for lvl, text in job.log[shown:]:
                if lvl in ('warning', 'error'):
                    print(f'  {lvl}: {text}', file=sys.stderr)
            shown = len(job.log)
            time.sleep(0.2)
    except KeyboardInterrupt:
        svc.queue.cancel(job)
        while job.state not in FINAL:
            svc.queue.poll()
            time.sleep(0.2)
    code = job.exit_code if job.exit_code is not None else P.FAILED
    return P.CANCELLED if code < 0 and job.cancel_requested else (P.CRASHED if code < 0 else code)


def main_replay(argv) -> int:
    """The `falcon replay` command: protocol lines on stdout; exit 0 ok, else the worker's code or 2."""
    ap = argparse.ArgumentParser(prog='falcon replay', description='import logs, replay them in CARLA with changes, '
                                                                    'and compare (docs/log-replay.md)')
    sub = ap.add_subparsers(dest='what', required=True)
    im = sub.add_parser('import', help='import a log: nuscenes|waymo|ros2|kitti_raw|falcon:PATH (or PATH)')
    im.add_argument('source')
    im.add_argument('--out', '-o', required=True, help='the log\'s folder (log.json)')
    im.add_argument('--scene')
    im.add_argument('--segment')
    im.add_argument('--channel')
    im.add_argument('--town', help='ROS bags: the CARLA town their map frame is in, when known')
    im.add_argument('--world-frame')
    im.add_argument('--ego-frame')
    im.add_argument('--topic', action='append', default=[], help='ROS bags: object topics to read (default all)')
    im.add_argument('--unknown-by-size', action='store_true')
    ma = sub.add_parser('match', help='where a log goes in a world, and how well its ego path fits the lanes')
    ma.add_argument('log')
    ma.add_argument('project')
    ma.add_argument('--world', default='')
    ru = sub.add_parser('run', help='replay a log in CARLA (a job)')
    ru.add_argument('project')
    ru.add_argument('--log', required=True)
    ru.add_argument('--out', '-o', required=True, help='the run folder')
    ru.add_argument('--world', default='')
    ru.add_argument('--rig', default='')
    ru.add_argument('--changes', help='a JSON file: the change list (core.logreplay.changes)')
    ru.add_argument('--seed', type=int, default=0)
    ru.add_argument('--fps', type=float, default=20.0)
    ru.add_argument('--driver', default='log', choices=('log', 'follower', 'plugin'))
    ru.add_argument('--policy', help='with --driver plugin: MODULE:CLASS')
    ru.add_argument('--t-start', type=float)
    ru.add_argument('--t-end', type=float)
    ru.add_argument('--anchor-spawn', type=int, default=0)
    ru.add_argument('--ego-offset', type=float, default=0.0)
    ru.add_argument('--allow-mismatch', action='store_true')
    ru.add_argument('--keep-server', action='store_true')
    ru.add_argument('--fake-carla', default='', help=argparse.SUPPRESS)
    co = sub.add_parser('compare', help='compare an original replay with changed ones')
    co.add_argument('--original', required=True)
    co.add_argument('--changed', nargs='+', required=True)
    co.add_argument('--out', '-o', required=True)
    co.add_argument('--project')
    co.add_argument('--model')
    co.add_argument('--synthetic-detector', action='store_true')
    co.add_argument('--camera')
    co.add_argument('--every', type=int, default=1)
    co.add_argument('--boot', type=int, default=1000)
    co.add_argument('--seed', type=int, default=0)
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return P.OK if e.code == 0 else P.FAILED
    em = P.Emitter()
    em.hello(f'replay {a.what}')
    try:
        if a.what == 'import':
            from . import KINDS, import_log
            kind, _, path = a.source.partition(':')
            if not path or kind not in KINDS:
                kind, path = None, a.source
            log = import_log(kind, path, scene=a.scene, segment=a.segment, channel=a.channel, town=a.town,
                             world_frame=a.world_frame, ego_frame=a.ego_frame, topics=a.topic or None,
                             unknown_by_size=a.unknown_by_size)
            p = log.save(a.out)
            c = log.counts()
            em.result('ok', summary={'log': p, 'map': log.map.get('kind'), 'map_name': log.map.get('name'), **c})
            print(f'{log.meta.get("name")}: {c["actors"]} actors {c["by_class"]}, {c["duration_s"]:.1f} s, ego path '
                  f'{c["ego_path_m"]:.0f} m; map {log.map.get("kind")} {log.map.get("name") or ""} → {p}',
                  file=sys.stderr)
            return P.OK
        if a.what == 'compare':
            from .compare import compare
            model, ws = None, ''
            if a.synthetic_detector:
                model = {'id': 'synthetic', 'name': 'Synthetic detector (not a model)', 'kind': 'detector',
                         'runtime': 'python', 'entry': 'falcon_autonomy.plugins.synthetic_detector:SyntheticDetector',
                         'licence': 'GPL-3.0-or-later'}
            elif a.model:
                from ..paths import workspace_for
                from ..project import Project
                if not a.project:
                    raise LogError('--model needs --project')
                proj = Project.load(a.project)
                model = proj.item('models', a.model)
                if model is None:
                    raise LogError(f'no model {a.model!r} in the project')
                ws = workspace_for(a.project)
            doc = compare(a.original, a.changed, model, ws, a.camera, a.every, a.boot, a.seed, a.out,
                          lambda f, t: em.progress(f, t))
            d0 = doc['pairs'][0]['difference']
            em.result('ok', summary={'pairs': len(doc['pairs']), 'min_ttc_s_diff': d0.get('min_ttc_s'),
                                     'min_distance_m_diff': d0.get('min_distance_m'),
                                     'collision_diff': d0.get('collision')})
            return P.OK
        from ..project import Project
        from .plan import make_plan, world_info
        proj = Project.load(a.project)
        world = a.world or proj.doc['worlds'][0]['id']
        winfo = world_info(proj, world)
        log_path = a.log if a.log.endswith('.json') else os.path.join(a.log, 'log.json')
        log = Log.load(log_path)
        if a.what == 'match':
            res = offline_match(log, winfo, proj.workspace)
            em.result('ok', summary={'placement': res['placement']['mode'],
                                     'verdict': (res.get('match') or {}).get('verdict'),
                                     'why': (res.get('match') or {}).get('why') or res.get('match_note')})
            print(json.dumps(res, indent=1), file=sys.stderr)
            return P.OK
        changes = []
        if a.changes:
            with open(a.changes, encoding='utf-8') as fh:
                changes = json.load(fh)
        plan = make_plan(log, winfo, fps=a.fps, changes=changes, seed=a.seed, driver=a.driver, policy=a.policy,
                         t_start=a.t_start, t_end=a.t_end, anchor={'spawn_index': a.anchor_spawn},
                         ego_offset_m=a.ego_offset, allow_mismatch=a.allow_mismatch, log_path=log_path)
        from ..jobs import JobQueue, SubprocessLauncher
        from ..settings import Settings
        from ...sim.service import CarlaService
        svc = CarlaService(Settings(), JobQueue(SubprocessLauncher()), proj.workspace, fake_carla=a.fake_carla)
        try:
            cfg = replay_config(proj, plan, world, a.rig or proj.doc['rigs'][0]['id'], a.out, svc.port,
                                svc.tm_port, float(svc.settings.get('render_budget_mpx')), proj.workspace)
            job = submit(svc, cfg)
            code = _wait(svc, job, em)
        finally:
            if not a.keep_server:
                svc.shutdown()
        run = {}
        try:
            with open(os.path.join(a.out, 'run.json'), encoding='utf-8') as fh:
                run = json.load(fh)
        except (OSError, ValueError):
            pass
        sc = run.get('scenario') or {}
        em.result(P.status_of(code), error=run.get('error'), run_json=os.path.join(a.out, 'run.json'),
                  summary={'steps': (run.get('performance') or {}).get('steps'),
                           'placement': plan['placement']['mode'], 'replay': sc.get('replay'),
                           'collision': ((sc.get('criticality') or {}).get('overall') or {}).get('collision')})
        if code != P.OK and run.get('error'):
            print(f'failed: {run["error"]}', file=sys.stderr)
        return code
    except (LogError, ValueError, OSError, KeyError) as e:
        print(f'failed: {e}', flush=True)
        return P.FAILED
