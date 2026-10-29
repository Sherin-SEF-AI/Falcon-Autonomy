"""Drive configs: what the drive worker runs, built from a project's world, rig and traffic profile (standard library).

The GUI and the CLI both call drive_config(), so a drive started from either is the same run. The config is written
into the run folder (drive.json) and its hash is in run.json, which is what `falcon reproduce` re-runs.
"""
from __future__ import annotations

import copy
import datetime as _dt
import os
import re

from . import weather as W
from .project import Project, ProjectError

DRIVE_SCHEMA = 'falcon.drive_config/1'
SMOKE_WALL_S = 30.0


def run_dir_name(label: str) -> str:
    stamp = _dt.datetime.now().strftime('%Y%m%d-%H%M%S')
    slug = re.sub(r'[^A-Za-z0-9_-]+', '-', label).strip('-').lower()[:40] or 'run'
    return f'{stamp}-{slug}'


def drive_config(project: Project, world_id: str, rig_id: str, out_dir: str, port: int, tm_port: int,
                 seconds: float = 10.0, fps: float = 20.0, seed: int = 0, traffic_id: str | None = None,
                 warmup_steps: int = 10, max_wall_s: float = 0.0, render_budget_mpx: float = 200.0,
                 host: str = '127.0.0.1', name: str = '', workspace: str = '') -> dict:
    """A drive config. A town world names its map; an imported world (kind osm or xodr) names its OpenDRIVE file,
    kept in the workspace (world['xodr'], relative to it), which the worker loads with generate_opendrive_world."""
    world = project.item('worlds', world_id)
    rig = project.item('rigs', rig_id)
    if world is None:
        raise ProjectError(f'no world {world_id!r} in the project')
    if rig is None:
        raise ProjectError(f'no rig {rig_id!r} in the project')
    xodr = ''
    if world['kind'] != 'town':
        ws = workspace or project.workspace
        xodr = os.path.join(ws, world['xodr']) if world.get('xodr') and ws else ''
        if not xodr or not os.path.isfile(xodr):
            raise ProjectError(f'world {world["name"]!r} has no converted OpenDRIVE file yet: import it on the Worlds '
                               'page first')
    tp_id = traffic_id or world.get('traffic_profile') or 'tp-default'
    traffic = project.item('traffic_profiles', tp_id) or {}
    from .traffic import check_handedness
    rule = (world.get('osm') or {}).get('handedness', 'RHT') if xodr else 'RHT'
    ok, why = check_handedness(traffic, rule)
    if not ok:
        raise ProjectError(why)
    cfg = {
        'schema': DRIVE_SCHEMA, 'name': name or f'{world["name"]} / {rig["name"]}',
        'host': host, 'port': int(port), 'tm_port': int(tm_port),
        'town': world.get('town', '') if not xodr else '', 'xodr': xodr,
        'opendrive': copy.deepcopy((world.get('osm') or {}).get('mesh') or {}),
        'handedness': rule,
        'weather': W.resolve(world.get('weather')),
        'weather_source': copy.deepcopy(world.get('weather') or {}),
        'rig': copy.deepcopy(rig), 'traffic': copy.deepcopy(traffic),
        'fps': float(fps), 'steps': max(int(round(seconds * fps)), 1), 'warmup_steps': int(warmup_steps),
        'max_wall_s': float(max_wall_s), 'seed': int(seed), 'render_budget_mpx': float(render_budget_mpx),
        'out': os.path.abspath(out_dir), 'project': {'name': project.doc['name'], 'world': world_id, 'rig': rig_id,
                                                     'traffic_profile': tp_id},
    }
    if xodr:                            # traffic at each road's limit from the file (workers.speedcontrol), or CARLA's own
        cfg['speed_limits'] = world.get('speed_limits', 'map')
        cfg['keep_traffic'] = bool(world.get('keep_traffic', True))   # traffic kept in it at roads that end
    if (world.get('sensor_physics') or {}).get('enabled'):      # the world's weather physics (docs/sensor-physics.md);
        cfg['sensor_physics'] = copy.deepcopy(world['sensor_physics'])      # absent otherwise: cache keys unchanged
    if xodr and (world.get('city') or {}).get('enabled'):      # buildings, trees and ground around the roads
        from .cityprops import attach as city_attach            # (core.cityprops); absent otherwise
        city_attach(cfg, world, workspace or getattr(project, 'workspace', '') or '')
    from .oddmonitor import attach                              # the workspace's ODD, monitored by the worker
    attach(cfg, workspace or getattr(project, 'workspace', '') or '', project)
    return cfg


WORLD_KEYS = ('speed_limits', 'keep_traffic', 'city', 'vehicle_lights')


def world_keys(d: dict) -> dict:
    """The world's own settings of a drive config that a worker's scene setup reads (scene.load_map, workers.lights):
    for configs built from a drive config's fields (gym, HIL), so they load the world as a drive does."""
    return {k: copy.deepcopy(d[k]) for k in WORLD_KEYS if k in d}


def smoke_config(project: Project, out_dir: str, port: int, tm_port: int, seed: int = 0) -> dict:
    """The first-run smoke drive: the project's first town and rig, 30 s of wall clock, a handful of cars."""
    world = next((w for w in project.doc['worlds'] if w['kind'] == 'town'), None)
    if world is None or not project.doc['rigs']:
        raise ProjectError('the smoke drive needs a town world and a rig in the project')
    cfg = drive_config(project, world['id'], project.doc['rigs'][0]['id'], out_dir, port, tm_port,
                       seconds=600.0, fps=20.0, seed=seed, max_wall_s=SMOKE_WALL_S, name='smoke drive')
    cfg['traffic'] = dict(cfg['traffic'], vehicles=min(int(cfg['traffic'].get('vehicles', 0)), 10))
    return cfg


def scenario_config(project: Project, scenario: dict, world_id: str, rig_id: str, out_dir: str, port: int,
                    tm_port: int, seed: int = 0, values: dict | None = None, fps: float = 20.0,
                    traffic_vehicles: int = 0, render_budget_mpx: float = 200.0, workspace: str = '') -> dict:
    """A drive running a scenario: sampled with the seed (or the given parameter values), the background traffic
    off unless asked, its length the scenario's."""
    from . import scenarios as SC
    conc = SC.sample(scenario, seed, values) if values is None or set(values) != set(scenario.get('params') or {}) \
        else dict(SC.concrete(scenario, values), seed=seed)
    cfg = drive_config(project, world_id, rig_id, out_dir, port, tm_port, seconds=float(conc.get('duration_s', 12)),
                       fps=fps, seed=seed, render_budget_mpx=render_budget_mpx, name=f'{scenario["name"]} (seed {seed})',
                       workspace=workspace)
    cfg['traffic'] = dict(cfg['traffic'], vehicles=int(traffic_vehicles), walkers=0)
    cfg['scenario'] = conc
    if conc.get('weather'):                   # a scenario's own weather (e.g. the monsoon variant) replaces the world's
        cfg['weather'] = W.resolve(conc['weather'])
        cfg['weather_source'] = {'scenario': conc.get('id'), **copy.deepcopy(conc['weather'])}
        if cfg.get('xodr') and float(cfg['weather'].get('precipitation_deposits', 0.0)) > 0:
            cfg.setdefault('notes', []).append(
                'puddles (precipitation_deposits) are not drawn on an imported OpenDRIVE world (measured on CARLA '
                '0.9.16: no visible change on a generated road, a large one on Town10HD); rain and wetness are')
    cfg['warmup_steps'] = 5
    cfg['reload_world'] = True            # bit-for-bit repetition needs the same initial world state
    return cfg
