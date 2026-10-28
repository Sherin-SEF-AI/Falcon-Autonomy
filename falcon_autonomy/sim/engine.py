"""The data engine's runner (standard library): a dataset spec's plan turned into queued drives, cached and resumable,
shared by the GUI and the CLI.

Each planned drive's run folder is <workspace>/datasets/<spec>/drives/<cache key[:16]>: its name is the drive's cache
key (core.datasets.drive_key), so a drive that already finished with that exact config is reused, and a run that was
cancelled or failed is run again — re-running a spec resumes it.
"""
from __future__ import annotations

import copy
import json
import os

from ..core import datasets as D
from ..core.budget import rig_mpx
from ..core.drive import drive_config
from ..core.provenance import read_json, write_json


def spec_dir(workspace: str, spec_id: str) -> str:
    return os.path.join(workspace, 'datasets', spec_id)


def finished_runs(workspace: str, spec_id: str) -> list:
    d = os.path.join(spec_dir(workspace, spec_id), 'drives')
    out = []
    if os.path.isdir(d):
        for name in sorted(os.listdir(d)):
            try:
                if read_json(os.path.join(d, name, 'run.json')).get('status') == 'ok':
                    out.append(os.path.join(d, name))
            except (OSError, ValueError):
                continue
    return out


def coverage_rows(workspace: str, spec: dict) -> list:
    return [D.coverage_of_run(r, spec.get('bins')) for r in finished_runs(workspace, spec['id'])]


def make_plan(project, spec: dict, throughput) -> dict:
    rows = coverage_rows(project.workspace, spec)
    have = D.coverage(spec, rows)['have']
    rig = project.item('rigs', spec['rig'])
    if rig is None:
        raise ValueError(f'the spec\'s rig {spec["rig"]!r} is not in the project')
    p = D.plan(spec, have, D.measured_yield(rows), throughput, rig_mpx(rig))
    p['have'] = have
    p['done'] = len(rows)
    return p


def drive_configs(project, spec: dict, plan: dict, port: int, tm_port: int, budget_mpx: float) -> list:
    """The planned drives as drive configs, each with its cache-keyed run folder; (config, cached: bool)."""
    out = []
    base = os.path.join(spec_dir(project.workspace, spec['id']), 'drives')
    for d in plan['drives']:
        cell = d['cell']
        cfg = drive_config(project, cell['world'], spec['rig'], '/pending', port, tm_port, spec['drive_s'],
                           spec['fps'], d['seed'], traffic_id=spec.get('traffic_profile'),
                           render_budget_mpx=budget_mpx, name=f'{spec["name"]} #{d["index"]}')
        from ..core import weather as W
        cfg['weather'] = W.resolve(cell['weather'])
        cfg['weather_source'] = copy.deepcopy(cell['weather'])
        cfg['record_every'] = int(spec.get('record_every', 1))
        cfg['dataset'] = {'spec': spec['id'], 'index': d['index'], 'cell': cell}
        key = D.drive_key(cfg)
        cfg['out'] = os.path.join(base, key[:16])
        cfg['cache_key'] = key
        cached = False
        try:
            cached = read_json(os.path.join(cfg['out'], 'run.json')).get('status') == 'ok'
        except (OSError, ValueError):
            pass
        out.append((cfg, cached))
    return out


def submit(service, project, spec: dict, plan: dict) -> dict:
    """Queue every planned drive that is not cached. Returns {'queued': [jobs], 'cached': n}."""
    cfgs = drive_configs(project, spec, plan, service.port, service.tm_port,
                         float(service.settings.get('render_budget_mpx')))
    jobs, cached = [], 0
    for cfg, hit in cfgs:
        if hit:
            cached += 1
            continue
        jobs.append(service.submit_drive(cfg, title=cfg['name'], kind='drive'))
    write_json(os.path.join(spec_dir(project.workspace, spec['id']), 'plan.json'),
               {'spec': spec, 'plan': plan, 'drives': [c['out'] for c, _ in cfgs]})
    return {'queued': jobs, 'cached': cached}


def versions(workspace: str, spec_id: str) -> list:
    d = os.path.join(spec_dir(workspace, spec_id), 'versions')
    if not os.path.isdir(d):
        return []
    out = []
    for f in sorted(os.listdir(d)):
        if f.endswith('.json'):
            with open(os.path.join(d, f), encoding='utf-8') as fh:
                out.append(json.load(fh))
    return out


def new_version(workspace: str, spec: dict, by: str, shares: dict, seed: int = 0) -> dict:
    """A versioned snapshot of the finished drives with a leak-free split by drive or by place (world)."""
    runs = finished_runs(workspace, spec['id'])
    rows = {r: D.coverage_of_run(r, spec.get('bins')) for r in runs}
    if by == 'place':
        worlds = sorted({rows[r].get('world') or '?' for r in runs})
        assign = D.split(worlds, shares, seed)
        splits = {k: [os.path.relpath(r, workspace) for r in runs if (rows[r].get('world') or '?') in v]
                  for k, v in assign.items()}
    else:
        rel = [os.path.relpath(r, workspace) for r in runs]
        w = {os.path.relpath(r, workspace): max(rows[r]['frames'], 1) for r in runs}
        splits = D.split(rel, shares, seed, w)
    n = len(versions(workspace, spec['id'])) + 1
    doc = D.make_version(spec, runs, splits, by, n, workspace)
    write_json(os.path.join(spec_dir(workspace, spec['id']), 'versions', f'v{n:03d}.json'), doc)
    return doc
