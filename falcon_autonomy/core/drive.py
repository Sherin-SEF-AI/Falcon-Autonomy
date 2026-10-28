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
    return {
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


def smoke_config(project: Project, out_dir: str, port: int, tm_port: int, seed: int = 0) -> dict:
    """The first-run smoke drive: the project's first town and rig, 30 s of wall clock, a handful of cars."""
    world = next((w for w in project.doc['worlds'] if w['kind'] == 'town'), None)
    if world is None or not project.doc['rigs']:
        raise ProjectError('the smoke drive needs a town world and a rig in the project')
    cfg = drive_config(project, world['id'], project.doc['rigs'][0]['id'], out_dir, port, tm_port,
                       seconds=600.0, fps=20.0, seed=seed, max_wall_s=SMOKE_WALL_S, name='smoke drive')
    cfg['traffic'] = dict(cfg['traffic'], vehicles=min(int(cfg['traffic'].get('vehicles', 0)), 10))
    return cfg
