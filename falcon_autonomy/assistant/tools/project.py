"""Project tools: read and edit the project's worlds, rigs, scenarios, traffic profiles, trigger rules, roadside units and
ODDs — the definitions the Build pages edit.

Every change is one core.project.Edit, built as the page builds it (the same helpers: core.weather, core.rig,
core.scenarios, core.traffic, core.triggers, core.v2x, core.mapedit, core.worlds, core.loop, core.safetycase), checked
against the project schema and the module's own validator before it is applied, and applied through the window's undo
stack (C.apply_edit). An item is replaced whole where a page sets several of its fields, so one tool call is one undo
step. Each acting tool's preview is that edit as a short diff.
"""
from __future__ import annotations

import copy
import os

from ..tool_api import ToolError, tool
from . import _common as C

SECTIONS = {'world': 'worlds', 'rig': 'rigs', 'scenario': 'scenarios', 'traffic_profile': 'traffic_profiles',
            'trigger_rule': 'trigger_rules', 'drive_set': 'drive_sets'}


def _validated(ctx, edit, extra=None):
    """The edit applied to a copy of the project must still match the schema (and extra(doc) -> [problems])."""
    from ...core.project import validate
    d = copy.deepcopy(C.doc(ctx))
    try:
        edit.apply(d)
    except (KeyError, IndexError, TypeError, ValueError) as e:
        raise ToolError(f'that edit does not fit the project: {e}') from e
    errs = validate(d) + (extra(d) if extra else [])
    if errs:
        raise ToolError('refused, the project would not be valid: ' + '; '.join(str(x) for x in errs[:8]))
    return edit


def _done(ctx, edit, **more) -> dict:
    out = C.apply_edit(ctx, edit)
    out.update(more)
    return out


# ================================================================================================= reading
@tool('project_overview', 'project', 'read',
      'The open project at a glance: its name and file, unsaved changes, and the ids and names of its worlds (kind, '
      'town or imported, weather preset), rigs (sensor count), scenarios (project ones; built-ins via '
      'project_scenarios), traffic profiles, trigger rules (enabled), roadside units, drive sets, registered models, '
      'the loop\'s model and ODD, and what it has recorded: how many runs (the newest five) and evaluations. Use the '
      'ids in the other tools.')
def project_overview(ctx):
    p = C.project(ctx)
    d = p.doc

    def rows(sec, f):
        return [f(x) for x in (d.get(sec) or [])][:40]
    return {
        'name': d.get('name'), 'file': p.path or '(not saved yet)', 'unsaved_changes': bool(p.dirty),
        'workspace': C.workspace(ctx) if p.path or ctx.win is not None else '',
        'worlds': rows('worlds', lambda w: {'id': w['id'], 'name': w.get('name'), 'kind': w.get('kind'),
                                            'town': w.get('town'), 'weather': (w.get('weather') or {}).get('preset'),
                                            'traffic_profile': w.get('traffic_profile')}),
        'rigs': rows('rigs', lambda r: {'id': r['id'], 'name': r.get('name'), 'vehicle': r.get('vehicle'),
                                        'sensors': len(r.get('sensors') or [])}),
        'scenarios': rows('scenarios', lambda s: {'id': s['id'], 'name': s.get('name')}),
        'traffic_profiles': rows('traffic_profiles', lambda t: {'id': t['id'], 'name': t.get('name'),
                                                                'vehicles': t.get('vehicles'),
                                                                'walkers': t.get('walkers')}),
        'trigger_rules': rows('trigger_rules', lambda r: {'id': r['id'], 'rule': C.clip(r.get('rule'), 120),
                                                          'enabled': r.get('enabled', True)}),
        'roadside_units': [{'id': r['id'], 'name': r.get('name'), 'world': r.get('world')}
                           for r in ((d.get('infrastructure') or {}).get('rsus') or [])][:40],
        'drive_sets': rows('drive_sets', lambda s: {'id': s['id'], 'name': s.get('name'), 'kind': s.get('kind')}),
        'models': rows('models', lambda m: {'id': m['id'], 'name': m.get('name'), 'kind': m.get('kind')}),
        'loop': {'model': (d.get('loop') or {}).get('model'), 'odd': ((d.get('loop') or {}).get('odd') or {}).get(
            'name') if (d.get('loop') or {}).get('odd') else None},
        'recorded': _recorded(ctx) if p.path or ctx.win is not None else None,
    }


def _recorded(ctx) -> dict:
    """What the project has made: its runs (newest few; private ones only counted) and evaluations."""
    import os
    from .runs import _entries, _row
    es, hidden = _entries(ctx)
    es = sorted(es, key=lambda e: e.get('started') or '', reverse=True)
    ev = os.path.join(C.workspace(ctx), 'evaluations')
    n_ev = sum(os.path.isfile(os.path.join(ev, x, 'evaluation.json')) for x in os.listdir(ev)) \
        if os.path.isdir(ev) else 0
    out = {'runs': len(es), 'latest_runs': [{k: v for k, v in _row(ctx, e).items()
                                             if k in ('id', 'name', 'status', 'world', 'frames', 'started')}
                                            for e in es[:5]],
           'evaluations': n_ev, 'more': 'runs_search for every run'}
    if hidden:
        out['private_runs_not_listed'] = hidden
    return out


@tool('project_get', 'project', 'read',
      'One project item in full by kind and id: a world, rig, scenario (a project one, or a built-in by its key), '
      'traffic profile, trigger rule, roadside unit (kind rsu), drive set or model; kind odd returns the loop\'s ODD '
      '(id ignored) and kind safety_case_odd the workspace\'s safety-case ODD (odd.json). Long lists inside are '
      'capped.',
      C.obj({'kind': C.s('what to get', enum=['world', 'rig', 'scenario', 'traffic_profile', 'trigger_rule', 'rsu',
                                              'drive_set', 'model', 'odd', 'safety_case_odd']),
             'id': C.s('its id (or a unique name); a built-in scenario by its key')}, ['kind']))
def project_get(ctx, kind, id=''):
    d = C.doc(ctx)
    if kind == 'odd':
        return {'odd': (d.get('loop') or {}).get('odd')}
    if kind == 'safety_case_odd':
        p = os.path.join(C.workspace(ctx), 'odd.json')
        return {'odd': C.read_json(C.checked(ctx, p)) if os.path.isfile(p) else None,
                'file': 'odd.json' if os.path.isfile(p) else '(none yet: project_odd_set creates it)'}
    if kind == 'rsu':
        rs = (d.get('infrastructure') or {}).get('rsus') or []
        r = next((x for x in rs if x['id'] == id), None)
        if r is None:
            raise ToolError(f'no roadside unit {id!r} (ids: {", ".join(x["id"] for x in rs) or "none"})')
        return {'rsu': r}
    if kind == 'scenario':
        from ...core import scenarios as SC
        if id in SC.LIBRARY:
            return {'scenario': SC.builtin(id), 'builtin': True}
    if kind == 'model':
        return {'model': C.find(ctx, 'models', id, 'model')[1]}
    return {kind: C.rounded(C.find(ctx, SECTIONS[kind], id, kind.replace('_', ' '))[1])}


@tool('project_scenarios', 'project', 'read',
      'Every scenario that can be run: the built-in library (key, name, what it tests, its parameters with their '
      'distributions) and the project\'s own (editable). Built-ins are copied with project_scenario_create before '
      'they can be edited.')
def project_scenarios(ctx):
    from ...core import scenarios as SC

    def brief(s, builtin):
        return {'id': s['id'], 'name': s.get('name'), 'builtin': builtin, 'about': C.clip(s.get('about'), 200),
                'duration_s': s.get('duration_s'), 'params': {k: {kk: v for kk, v in p.items() if kk != 'about'}
                                                              for k, p in (s.get('params') or {}).items()}}
    out = [brief(SC.builtin(k), True) for k in SC.LIBRARY]
    out += [brief(s, False) for s in C.doc(ctx).get('scenarios') or []]
    return C.rounded(C.capped(out, 40))


@tool('project_options', 'project', 'read',
      'The choices the project editors offer, to build valid edits: CARLA towns, weather presets and slider ranges, '
      'sensor types with their defaults and vehicles, scenario actor kinds, timeline triggers and actions, parameter '
      'distributions, trigger-rule repeats/contexts and syntax, roadside-unit presets, traffic behaviour keys and '
      'lateral-filtering choices, map-edit operations.',
      C.obj({'topic': C.s('one topic, or all', enum=['all', 'worlds', 'rigs', 'scenarios', 'triggers', 'v2x',
                                                     'traffic', 'map_edits'])}))
def project_options(ctx, topic='all'):
    out = {}
    if topic in ('all', 'worlds'):
        from ...core import weather as W
        from ...core.worlds import TOWNS
        out['worlds'] = {'towns': list(TOWNS), 'weather_presets': list(W.PRESETS),
                         'weather_sliders': {k: {'lo': v[0], 'hi': v[1], 'unit': v[2], 'label': v[3]}
                                             for k, v in W.SLIDERS.items()},
                         'speed_limits': ['map', 'carla'],
                         'sensor_physics_keys': ['enabled', 'lidar', 'radar', 'camera']}
    if topic in ('all', 'rigs'):
        from ...core import faults as FL
        from ...core import rig as R
        out['rigs'] = {'sensor_types': list(R.SENSOR_TYPES), 'defaults': R.DEFAULTS, 'vehicles': list(R.VEHICLES),
                       'camera_labels': ['semantic', 'instance', 'depth', 'boxes2d', 'boxes3d', 'flow'],
                       'lidar_labels': ['semantic', 'instance'], 'timed_fault_kinds': list(FL.TIMED)}
    if topic in ('all', 'scenarios'):
        from ...core import scenarios as SC
        out['scenarios'] = {'actor_kinds': list(SC.KINDS), 'triggers': {k: list(v) for k, v in SC.TRIGGERS.items()},
                            'actions': SC.ACTIONS, 'distributions': list(SC.DISTS),
                            'actor_fields': ['id', 'kind', 'place{lane, ahead_m, lateral_m}', 'heading', 'speed_kmh',
                                             'start']}
    if topic in ('all', 'triggers'):
        from ...core import triggers as T
        out['triggers'] = {'repeats': list(T.REPEATS), 'contexts': list(T.CONTEXTS), 'syntax': C.clip(T.help_text(),
                                                                                                         2500)}
    if topic in ('all', 'v2x'):
        from ...core import v2x as V
        out['v2x'] = {'presets': list(V.PRESETS), 'height_m_default': V.HEIGHT_M, 'cpm_defaults': V.CPM_DEFAULTS}
    if topic in ('all', 'traffic'):
        from ...core import traffic as TR
        out['traffic'] = {'presets': list(TR.PRESETS), 'handedness': ['map', 'left', 'right'],
                          'behaviour_keys': ['distance_to_leading_m', 'speed_spread_pct', 'speed_difference_pct',
                                             'ignore_lights_pct', 'ignore_signs_pct', 'random_left_lanechange_pct',
                                             'random_right_lanechange_pct', 'walker_crossing_factor',
                                             'auto_lane_change'],
                          'lateral_filtering': {'vehicles': list(TR.LATERAL_KINDS),
                                                'controllers': list(TR.LATERAL_CONTROLLERS),
                                                'defaults': TR.LATERAL_DEFAULTS}}
    if topic in ('all', 'map_edits'):
        from ...core import mapedit as ME
        out['map_edits'] = {'ops': list(ME.OPS), 'breakers': list(ME.BREAKERS), 'max_lanes': ME.MAX_LANES,
                            'speed_kmh': list(ME.SPEED_KMH),
                            'shapes': {'highway': {'way': 'id', 'value': 'type'},
                                       'oneway': {'way': 'id', 'value': 'no|yes|-1'},
                                       'lanes': {'way': 'id', 'forward': 'n', 'backward': 'n'},
                                       'maxspeed': {'way': 'id', 'kmh': 'n or null'}, 'remove': {'way': 'id'},
                                       'signal': {'node': 'id', 'on': 'bool'}, 'remove_feature': {'node': 'id'},
                                       'speed_breaker': {'way': 'id', 'lat': 'deg', 'lon': 'deg',
                                                         'value': 'bump|hump|table'}}}
    return out


# ================================================================================================= worlds
def _world_add(ctx, town='Town10HD_Opt', name='', weather_preset='ClearNoon'):
    from ...core import weather as W
    from ...core.project import Edit, new_id
    from ...core.worlds import TOWNS
    if town not in TOWNS:
        raise ToolError(f'{town!r} is not one of the base package\'s towns: {", ".join(TOWNS)}')
    if weather_preset not in W.PRESETS:
        raise ToolError(f'no weather preset {weather_preset!r} (project_options lists them)')
    d = C.doc(ctx)
    w = {'id': new_id('world'), 'name': name or f'{town.replace("_Opt", "")} ({len(d["worlds"]) + 1})',
         'kind': 'town', 'town': town, 'weather': {'preset': weather_preset},
         'traffic_profile': d['traffic_profiles'][0]['id'] if d['traffic_profiles'] else 'tp-default'}
    return _validated(ctx, Edit('insert', ['worlds'], w, text=f'Add the world {w["name"]}'))


_WORLD_ADD = C.obj({'town': C.s('a CARLA town (project_options topic worlds)'), 'name': C.s('its name'),
                    'weather_preset': C.s('a weather preset, default ClearNoon')})


@tool('project_world_add', 'project', 'act',
      'Add a CARLA town as a world (Worlds ▸ Add town), with a weather preset and the project\'s first traffic '
      'profile. Returns the new world\'s id. Real places come from project_world_import_osm.', _WORLD_ADD,
      preview=lambda ctx, **a: C.edit_preview(ctx, _world_add(ctx, **a)))
def project_world_add(ctx, **a):
    e = _world_add(ctx, **a)
    return _done(ctx, e, world=e.value['id'])


def _world_set(ctx, world, name=None, town=None, traffic_profile=None, weather_preset=None, weather=None,
               speed_limits=None, keep_traffic=None, city_enabled=None, sensor_physics=None):
    from ...core import sensorphysics as SP
    from ...core import weather as W
    from ...core.project import Edit
    i, w = C.find(ctx, 'worlds', world, 'world')
    nw = copy.deepcopy(w)
    if name is not None:
        nw['name'] = name or 'World'
    if town is not None:
        if w.get('kind') != 'town':
            raise ToolError('only a town world has a town; an imported world keeps its OpenDRIVE file')
        nw['town'] = town
    if traffic_profile is not None:
        C.find(ctx, 'traffic_profiles', traffic_profile, 'traffic profile')
        nw['traffic_profile'] = traffic_profile
    if weather_preset is not None:                       # a preset replaces the overrides (Worlds ▸ weather preset)
        if weather_preset not in W.PRESETS:
            raise ToolError(f'no weather preset {weather_preset!r} (project_options lists them)')
        nw['weather'] = {'preset': weather_preset}
    if weather:
        wt = dict(nw.get('weather') or {'preset': W.DEFAULT_PRESET})
        for k, v in weather.items():
            if k not in W.SLIDERS:
                raise ToolError(f'{k!r} is not a weather slider ({", ".join(W.SLIDERS)})')
            lo, hi = W.SLIDERS[k][:2]
            if not lo <= float(v) <= hi:
                raise ToolError(f'{k} must be within {lo:g}…{hi:g} {W.SLIDERS[k][2]}')
            wt[k] = round(float(v), 3)
        nw['weather'] = wt
    if speed_limits is not None or keep_traffic is not None or city_enabled is not None:
        if w.get('kind') == 'town':
            raise ToolError('speed limits, kept traffic and the city apply to an imported world only')
    if speed_limits is not None:
        nw['speed_limits'] = speed_limits
    if keep_traffic is not None:
        nw['keep_traffic'] = bool(keep_traffic)
    if city_enabled is not None:
        if not (w.get('city') or {}).get('plan'):
            raise ToolError('this world has no city built yet (Worlds ▸ City ▸ Build)')
        nw['city'] = dict(w['city'], enabled=bool(city_enabled))
    if sensor_physics is not None:
        block = {k: SP.DEFAULTS[k] for k in ('enabled', 'lidar', 'radar', 'camera')}
        block.update(w.get('sensor_physics') or {})
        for k, v in sensor_physics.items():
            block[k] = bool(v)
        nw['sensor_physics'] = block
    return _validated(ctx, Edit('set', ['worlds', i], nw, text=f'Edit the world {w["name"]}'))


_WORLD_SET = C.obj({
    'world': C.s('the world id'), 'name': C.s('a new name'), 'town': C.s('another town (town worlds only)'),
    'traffic_profile': C.s('a traffic profile id'),
    'weather_preset': C.s('a weather preset (replaces any slider overrides)'),
    'weather': {'type': 'object', 'additionalProperties': {'type': 'number'},
                'description': 'slider overrides, e.g. {"fog_density": 40} (sliders and ranges: project_options)'},
    'speed_limits': C.s('imported worlds: map (each road\'s limit from the file) or carla', enum=['map', 'carla']),
    'keep_traffic': C.b('imported worlds: keep traffic in at roads that end'),
    'city_enabled': C.b('imported worlds: the built city props on or off'),
    'sensor_physics': {'type': 'object', 'additionalProperties': {'type': 'boolean'},
                       'description': 'weather physics flags: enabled, lidar, radar, camera'}}, ['world'])


@tool('project_world_set', 'project', 'act',
      'Change a world as the Worlds page does: name, town, traffic profile, weather (a preset, and/or slider '
      'overrides such as fog_density or sun_altitude_angle), and for an imported world its speed-limit mode, kept '
      'traffic and city props on/off; sensor weather physics on/off. One undoable edit.', _WORLD_SET,
      preview=lambda ctx, **a: C.edit_preview(ctx, _world_set(ctx, **a)))
def project_world_set(ctx, **a):
    return _done(ctx, _world_set(ctx, **a))


def _osm_source(ctx, source):
    ws = C.workspace(ctx)
    if C._bad_name(source):
        raise ToolError(f'{source!r} is not a file name')
    p = os.path.join(ws, 'osm', source)
    if not os.path.isfile(p):
        have = sorted(f for f in os.listdir(os.path.join(ws, 'osm')) if f.endswith(('.osm', '.pbf'))) \
            if os.path.isdir(os.path.join(ws, 'osm')) else []
        raise ToolError(f'no {source!r} in the workspace\'s osm/ folder (there: {", ".join(have) or "nothing"}); put '
                        'an .osm or .osm.pbf file there, or use Worlds ▸ Import place')
    if not p.endswith(('.osm', '.osm.pbf', '.pbf')):
        raise ToolError('an OSM import reads .osm or .osm.pbf files')
    return C.checked(ctx, p)


def _osm_cfg(ctx, source, lat=None, lon=None, size_m=800.0, bbox=None, left_hand=True, lane_width=3.35,
             traffic_lights=True):
    from ...core import osm as O
    from ...core.project import new_id
    from ...core.worlds import import_config, world_dir
    src = _osm_source(ctx, source)
    if bbox:
        if len(bbox) != 4:
            raise ToolError('bbox is [south, west, north, east] in degrees')
        box = tuple(float(x) for x in bbox)
    elif lat is not None and lon is not None:
        box = O.bbox_around(float(lat), float(lon), float(size_m))
    else:
        box = None
    wid = new_id('world')
    cfg = import_config(src, world_dir(C.workspace(ctx), wid), box, bool(left_hand),
                        {'lane_width': float(lane_width), 'traffic_lights': bool(traffic_lights)})
    return wid, cfg


_OSM = C.obj({'source': C.s('an .osm/.osm.pbf file name in the workspace\'s osm/ folder'),
              'lat': C.n('area centre latitude'), 'lon': C.n('area centre longitude'),
              'size_m': C.n('square side in metres (default 800)', minimum=100, maximum=10000),
              'bbox': C.arr(C.n('degrees'), '[south, west, north, east] instead of a centre', minItems=4, maxItems=4),
              'left_hand': C.b('left-hand traffic (default true)'),
              'lane_width': C.n('metres (default 3.35)'), 'traffic_lights': C.b('default true')}, ['source'])


@tool('project_world_import_osm', 'project', 'act',
      'Import a real place from an OpenStreetMap file in the workspace\'s osm/ folder (Worlds ▸ Import place): a '
      'job converts it to OpenDRIVE in CARLA\'s Python (no server). When the job is done, project_world_add_imported '
      'with the returned world id adds the world to the project. The whole file is imported without a centre/bbox.',
      _OSM, estimate=lambda ctx, **a: {'seconds': None, 'what': 'depends on the area: measured imports took '
                                                                'seconds to a few minutes'},
      preview=lambda ctx, **a: 'queue an OSM import of osm/' + a.get('source', '') + ' (worker osm_import)')
def project_world_import_osm(ctx, **a):
    wid, cfg = _osm_cfg(ctx, **a)
    job = C.need_service(ctx).submit_osm_import(cfg)
    ctx.emit('job_queued', {'job': job.id, 'title': job.spec.title})
    return {'job': job.id, 'title': job.spec.title, 'world': wid,
            'next': 'jobs_wait for the job, then project_world_add_imported with this world id'}


def _imported(ctx, world):
    from ...core.project import Edit
    from ...core.worlds import edited_world, imported_world, world_dir
    if C._bad_name(world) or '/' in world:
        raise ToolError(f'{world!r} is not a world id')
    ws = C.workspace(ctx)
    i, w = None, None
    for k, x in enumerate(C.doc(ctx)['worlds']):
        if x['id'] == world:
            i, w = k, x
    d = world_dir(ws, world)
    if w is None:                                         # a new import: worlds/<id>/import.json
        cfg = C.read_json(os.path.join(C.checked(ctx, d), 'import-config.json')) if os.path.isdir(d) else None
        rep = C.read_json(os.path.join(d, 'import.json'))
        if not cfg or not rep or not os.path.isfile(os.path.join(d, 'world.xodr')):
            raise ToolError(f'no finished import for {world!r} (its job must be done)')
        return _validated(ctx, Edit('insert', ['worlds'], imported_world(C.doc(ctx), world, cfg, rep),
                                    text=f'Import {os.path.basename(cfg["source"])}'))
    # an existing world: its newest applied map edits (worlds/<id>/edit-<hash>/)
    if not os.path.isdir(d):
        raise ToolError(f'{world!r} has no import folder')
    edits = sorted((e for e in os.listdir(C.checked(ctx, d)) if e.startswith('edit-')
                    and os.path.isfile(os.path.join(d, e, 'world.xodr'))),
                   key=lambda e: os.path.getmtime(os.path.join(d, e)))
    if not edits:
        raise ToolError(f'{w["name"]} is in the project already and has no applied map edits waiting')
    out = os.path.join(d, edits[-1])
    cfg = C.read_json(os.path.join(out, 'import-config.json')) or {}
    rep = C.read_json(os.path.join(out, 'import.json')) or {}
    if 'edits' not in cfg:
        raise ToolError('the newest re-conversion has no edit list')
    return _validated(ctx, Edit('set', ['worlds', i], edited_world(w, cfg, rep, ws),
                                text=f'Apply map edits to {w["name"]}'))


@tool('project_world_add_imported', 'project', 'act',
      'Finish an import: add the world a finished project_world_import_osm job converted (by its world id), or, for '
      'a world already in the project, take its newest finished map-edit re-conversion (jobs_apply_map_edits) into '
      'the project. One undoable edit.', C.obj({'world': C.s('the world id the import or edit job returned')},
                                               ['world']),
      preview=lambda ctx, **a: C.edit_preview(ctx, _imported(ctx, **a)))
def project_world_add_imported(ctx, world):
    e = _imported(ctx, world)
    return _done(ctx, e, world=world)


def _map_edit(ctx, world, edit=None, drop=None):
    from ...core import mapedit as ME
    from ...core.project import Edit
    i, w = C.find(ctx, 'worlds', world, 'world')
    if w.get('kind') != 'osm':
        raise ToolError('map edits apply to a world imported from OpenStreetMap')
    edits = list((w.get('osm') or {}).get('edits') or [])
    if drop is not None:
        if not 0 <= int(drop) < len(edits):
            raise ToolError(f'there are {len(edits)} edits (0-based index)')
        text = f'{w["name"]}: drop "{ME.describe(edits[int(drop)])}"'
        new = edits[:int(drop)] + edits[int(drop) + 1:]
    elif edit:
        if edit.get('op') not in ME.OPS:
            raise ToolError(f'op must be one of {", ".join(ME.OPS)}')
        data = _osm_data(ctx, w)
        errs = ME.check_edit(data, edit) if data is not None else []
        if errs:
            raise ToolError('refused by the map editor\'s check: ' + '; '.join(errs[:5]))
        new = ME.upsert(edits, edit)
        text = f'{w["name"]}: {ME.describe(edit, data)}'
    else:
        raise ToolError('give an edit or drop')
    return _validated(ctx, Edit('set', ['worlds', i, 'osm', 'edits'], new, text=text))


def _osm_data(ctx, w):
    """The world's OSM source read as the map editor reads it (None when the source file is gone)."""
    from ...core import mapedit as ME
    from ...core import osm as O
    ws = C.workspace(ctx)
    folder = os.path.join(ws, os.path.dirname(w['xodr']))
    try:
        path, bbox, _what = ME.source_for(w, folder)
    except (OSError, ValueError):
        return None
    if not path or not os.path.isfile(path):
        return None
    from ..privacy import guard
    guard(path)                       # an OSM file of a dashcam route is private (its import.json says so)
    return O.read_osm(path, bbox=tuple(bbox) if bbox else None)


@tool('project_map_edit', 'project', 'act',
      'Add (or replace, for the same road or node) one map edit of an imported world — a road\'s type, direction, '
      'lanes, speed limit, removal, a signal, a speed breaker (shapes: project_options topic map_edits; way and node '
      'ids are OpenStreetMap\'s) — checked against the world\'s OSM data as the map editor does; or drop one by its '
      'index. The world changes once jobs_apply_map_edits re-converts it.',
      C.obj({'world': C.s('the imported world id'),
             'edit': {'type': 'object', 'description': 'one edit: {"op": …, "way"|"node": id, …}',
                      'properties': {'op': {'type': 'string'}}, 'required': ['op']},
             'drop': C.i('index of an edit to remove instead', minimum=0)}, ['world']),
      preview=lambda ctx, **a: C.edit_preview(ctx, _map_edit(ctx, **a)))
def project_map_edit(ctx, **a):
    return _done(ctx, _map_edit(ctx, **a))


# ================================================================================================= rigs
def _rig_add(ctx, name='', copy_of=None):
    from ...core.project import Edit, default_rig, new_id
    d = C.doc(ctx)
    r = copy.deepcopy(C.find(ctx, 'rigs', copy_of, 'rig')[1]) if copy_of else default_rig()
    r['id'] = new_id('rig')
    r['name'] = name or (f'{r["name"]} (copy)' if copy_of else f'Rig {len(d["rigs"]) + 1}')
    return _validated(ctx, Edit('insert', ['rigs'], r, text=f'Add the rig {r["name"]}'))


@tool('project_rig_add', 'project', 'act',
      'Add a rig: the default one (a 1280×720 front camera on a Lincoln MKZ) or a copy of an existing rig. Returns '
      'its id; add sensors with project_sensor_add.',
      C.obj({'name': C.s('its name'), 'copy_of': C.s('a rig id to copy')}),
      preview=lambda ctx, **a: C.edit_preview(ctx, _rig_add(ctx, **a)))
def project_rig_add(ctx, **a):
    e = _rig_add(ctx, **a)
    return _done(ctx, e, rig=e.value['id'])


def _rig_check(rig_index):
    def chk(d):
        from ...core import rig as R
        return R.validate_rig(d['rigs'][rig_index]) if rig_index is not None and rig_index < len(d['rigs']) else []
    return chk


def _rig_set(ctx, rig, name=None, vehicle=None):
    from ...core import rig as R
    from ...core.project import Edit
    i, r = C.find(ctx, 'rigs', rig, 'rig')
    nr = copy.deepcopy(r)
    if name is not None:
        nr['name'] = name
    if vehicle is not None:
        if vehicle not in R.VEHICLES:
            raise ToolError(f'vehicle must be one of {", ".join(R.VEHICLES)}')
        nr['vehicle'] = vehicle
    return _validated(ctx, Edit('set', ['rigs', i], nr, text=f'Edit the rig {r["name"]}'), _rig_check(i))


@tool('project_rig_set', 'project', 'act', 'Rename a rig or change its vehicle (project_options topic rigs lists '
      'the vehicles).', C.obj({'rig': C.s('the rig id'), 'name': C.s('a new name'),
                               'vehicle': C.s('a vehicle blueprint')}, ['rig']),
      preview=lambda ctx, **a: C.edit_preview(ctx, _rig_set(ctx, **a)))
def project_rig_set(ctx, **a):
    return _done(ctx, _rig_set(ctx, **a))


def _apply_fields(s: dict, fields: dict) -> dict:
    """Set a sensor's fields; dotted keys reach into blocks (lens.fov), as Rigs ▸ sensor fields does."""
    from ...core import faults as FL
    for k, v in (fields or {}).items():
        if k in ('id', 'type'):
            raise ToolError(f'a sensor\'s {k} cannot be changed (add a new sensor instead)')
        if k == 'faults':
            errs = FL.validate(v or [])
            if errs:
                raise ToolError('faults: ' + '; '.join(errs[:5]))
        cur = s
        parts = k.split('.')
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
            if not isinstance(cur, dict):
                raise ToolError(f'{k}: {p} is not a block')
        cur[parts[-1]] = v
    return s


def _sensor_add(ctx, rig, type, name='', pose=None, fields=None):
    from ...core import rig as R
    from ...core.project import Edit, new_id
    if type not in R.SENSOR_TYPES:
        raise ToolError(f'type must be one of {", ".join(R.SENSOR_TYPES)}')
    i, r = C.find(ctx, 'rigs', rig, 'rig')
    base = type.split('.')[-1]
    n = sum(1 for s in r['sensors'] if s['type'] == type) + 1
    p = {'z': 1.6, 'x': 1.0 if type.startswith('camera') else 0.0}
    p.update({k: float(v) for k, v in (pose or {}).items()})
    s = R.with_defaults(R.new_sensor(type, new_id(base), name or f'{base}_{n}', p))
    s = _apply_fields(s, fields)
    return _validated(ctx, Edit('insert', ['rigs', i, 'sensors'], s, text=f'Add {s["name"]} to {r["name"]}'),
                      _rig_check(i))


_POSE = C.obj({k: C.n(k + (' (m)' if k in 'xyz' else ' (deg)')) for k in ('x', 'y', 'z', 'roll', 'pitch', 'yaw')},
              description='mounting pose on the vehicle (CARLA axes: x forward, y right, z up)')


@tool('project_sensor_add', 'project', 'act',
      'Add a sensor to a rig (Rigs ▸ Add): a pinhole or fisheye camera, lidar, radar, GNSS, IMU or a microphone '
      'array (audio_array), with the type\'s defaults (core.rig.with_defaults), a pose, and any fields to set '
      '(e.g. width, height, fov, labels ["semantic","boxes2d"], channels, range, mics, radius; dotted keys like '
      'lens.fov). The rig is validated as the editor validates it.',
      C.obj({'rig': C.s('the rig id'), 'type': C.s('sensor type', enum=['camera.pinhole', 'camera.fisheye', 'lidar',
                                                                          'radar', 'gnss', 'imu', 'audio_array']),
             'name': C.s('its name'), 'pose': _POSE,
             'fields': {'type': 'object', 'description': 'fields to set (type defaults: project_options topic rigs)'}},
            ['rig', 'type']),
      preview=lambda ctx, **a: C.edit_preview(ctx, _sensor_add(ctx, **a)))
def project_sensor_add(ctx, **a):
    e = _sensor_add(ctx, **a)
    return _done(ctx, e, sensor=e.value['id'])


def _find_sensor(ctx, rig, sensor):
    i, r = C.find(ctx, 'rigs', rig, 'rig')
    for j, s in enumerate(r['sensors']):
        if s['id'] == sensor or s.get('name') == sensor:
            return i, r, j, s
    raise ToolError(f'no sensor {sensor!r} on {r["name"]} (ids: {", ".join(s["id"] for s in r["sensors"])})')


def _sensor_set(ctx, rig, sensor, name=None, pose=None, fields=None):
    from ...core.project import Edit
    i, r, j, s = _find_sensor(ctx, rig, sensor)
    ns = copy.deepcopy(s)
    if name is not None:
        ns['name'] = name or s['id']
    if pose:
        ns['pose'] = dict(ns['pose'], **{k: float(v) for k, v in pose.items()})
    ns = _apply_fields(ns, fields)
    return _validated(ctx, Edit('set', ['rigs', i, 'sensors', j], ns, text=f'{r["name"]}: edit {s["name"]}'),
                      _rig_check(i))


@tool('project_sensor_set', 'project', 'act',
      'Change a sensor of a rig: its name, pose (only the given axes) and fields (resolution, fov, labels, rate_hz, '
      'lidar settings, noise, timed faults [{sensor kind onset …}], weather_physics, mics/radius for an audio '
      'array; dotted keys reach into blocks). Validated with the rig checks; one undoable edit.',
      C.obj({'rig': C.s('the rig id'), 'sensor': C.s('the sensor id'), 'name': C.s('a new name'), 'pose': _POSE,
             'fields': {'type': 'object', 'description': 'fields to set'}}, ['rig', 'sensor']),
      preview=lambda ctx, **a: C.edit_preview(ctx, _sensor_set(ctx, **a)))
def project_sensor_set(ctx, **a):
    return _done(ctx, _sensor_set(ctx, **a))


# ================================================================================================= scenarios
def _scn_check(idx):
    def chk(d):
        from ...core import scenarios as SC
        return SC.validate(d['scenarios'][idx]) if idx is not None and idx < len(d['scenarios']) else []
    return chk


def _scenario_create(ctx, copy_of, name=''):
    from ...core import scenarios as SC
    from ...core.project import Edit, new_id
    src = SC.builtin(copy_of) if copy_of in SC.LIBRARY else C.find(ctx, 'scenarios', copy_of, 'scenario')[1]
    c = copy.deepcopy(src)
    c.update(id=new_id('scn'), name=name or src['name'] + ' (copy)', builtin=False)
    c.setdefault('schema', SC.SCENARIO_SCHEMA)
    return _validated(ctx, Edit('insert', ['scenarios'], c, text=f'Add the scenario {c["name"]}'),
                      _scn_check(len(C.doc(ctx)['scenarios'])))


@tool('project_scenario_create', 'project', 'act',
      'Create an editable project scenario as a copy of a built-in (by key, e.g. lead_brake) or of a project '
      'scenario (Scenarios ▸ Copy). Returns its id for project_scenario_set.',
      C.obj({'copy_of': C.s('a built-in key or a project scenario id'), 'name': C.s('its name')}, ['copy_of']),
      preview=lambda ctx, **a: C.edit_preview(ctx, _scenario_create(ctx, **a)))
def project_scenario_create(ctx, **a):
    e = _scenario_create(ctx, **a)
    return _done(ctx, e, scenario=e.value['id'])


def _scenario_set(ctx, scenario, name=None, about=None, duration_s=None, params=None, remove_params=None,
                  actors=None, add_actor=None, timeline=None, add_event=None, ego=None, weather=None,
                  pass_criteria=None):
    from ...core import scenarios as SC
    from ...core.project import Edit
    i, s = C.find(ctx, 'scenarios', scenario, 'scenario')
    ns = copy.deepcopy(s)
    for k, v in (('name', name), ('about', about), ('duration_s', duration_s), ('ego', ego), ('weather', weather),
                 ('pass', pass_criteria), ('actors', actors), ('timeline', timeline)):
        if v is not None:
            ns[k] = v
    if params:
        ns['params'] = dict(ns.get('params') or {}, **params)
    for k in remove_params or ():
        (ns.get('params') or {}).pop(k, None)
    if add_actor:
        a = {'id': f'actor{len(ns.get("actors") or []) + 1}', 'kind': 'car', 'place': {'lane': 'ego', 'ahead_m': 30.0},
             'heading': 'lane', 'speed_kmh': 0, 'start': 'stopped'}
        a.update(add_actor)
        ns['actors'] = list(ns.get('actors') or []) + [a]
    if add_event:
        ns['timeline'] = list(ns.get('timeline') or []) + [add_event]
    errs = SC.validate(ns)
    if errs:
        raise ToolError('refused by the scenario check: ' + '; '.join(errs[:8]))
    return _validated(ctx, Edit('set', ['scenarios', i], ns, text=f'Edit the scenario {s["name"]}'))


@tool('project_scenario_set', 'project', 'act',
      'Edit a project scenario (Scenarios page): name, about, duration, parameters (merged by name; each {dist: '
      'uniform|normal|choice|fixed, lo, hi | mean, sd | choices | value, unit}), parameters to remove, the actors '
      '(whole list, or add_actor with {id, kind, place{lane, ahead_m, lateral_m}, heading, speed_kmh, start}), the '
      'timeline (whole list, or add_event {trigger{type time|distance|ahead …}, actions[{actor, do …}]}), the ego, '
      'weather and pass criteria. Values may be "$param" expressions. Checked with core.scenarios.validate. A '
      'scenario has no world: the world is chosen when it is run (jobs_scenario_run).',
      C.obj({'scenario': C.s('the project scenario id'), 'name': C.s('name'), 'about': C.s('what it tests'),
             'duration_s': C.n('length in seconds', minimum=1),
             'params': {'type': 'object', 'description': 'parameters to add or replace, by name'},
             'remove_params': C.arr(C.s('name'), 'parameters to remove'),
             'actors': C.arr({'type': 'object'}, 'the whole actor list'),
             'add_actor': {'type': 'object', 'description': 'one actor to append (defaults filled in)'},
             'timeline': C.arr({'type': 'object'}, 'the whole timeline'),
             'add_event': {'type': 'object', 'description': 'one timeline event to append'},
             'ego': {'type': 'object', 'description': 'the ego block {speed_kmh, driver, route}'},
             'weather': {'type': 'object', 'description': 'the scenario\'s own weather (replaces the world\'s)'},
             'pass_criteria': {'type': 'object', 'description': '{no_collision, min_ttc_s}'}}, ['scenario']),
      preview=lambda ctx, **a: C.edit_preview(ctx, _scenario_set(ctx, **a)))
def project_scenario_set(ctx, **a):
    return _done(ctx, _scenario_set(ctx, **a))


# ================================================================================================= traffic
def _traffic_add(ctx, preset='tp-india-mixed', copy_of=None, name=''):
    from ...core import traffic as TR
    from ...core.project import Edit, new_id
    if copy_of:
        tp = copy.deepcopy(C.find(ctx, 'traffic_profiles', copy_of, 'traffic profile')[1])
        tp['id'] = new_id('tp')
    else:
        if preset not in TR.PRESETS:
            raise ToolError(f'preset must be one of {", ".join(TR.PRESETS)}')
        tp = TR.preset(preset)
        if C.project(ctx).item('traffic_profiles', tp['id']) is not None:
            tp['id'] = new_id('tp')
    if name:
        tp['name'] = name
    return _validated(ctx, Edit('insert', ['traffic_profiles'], tp, text=f'Add the traffic profile {tp["name"]}'))


@tool('project_traffic_add', 'project', 'act',
      'Add a traffic profile: a preset (tp-india-mixed: Indian mixed traffic with two-wheelers filtering between '
      'lanes, left-hand; tp-default: CARLA\'s defaults) or a copy of a profile. Returns its id.',
      C.obj({'preset': C.s('tp-india-mixed or tp-default'), 'copy_of': C.s('a profile id to copy'),
             'name': C.s('its name')}),
      preview=lambda ctx, **a: C.edit_preview(ctx, _traffic_add(ctx, **a)))
def project_traffic_add(ctx, **a):
    e = _traffic_add(ctx, **a)
    return _done(ctx, e, traffic_profile=e.value['id'])


def _traffic_set(ctx, profile, name=None, vehicles=None, walkers=None, handedness=None, fleet=None, behaviour=None,
                 lateral_filtering=None):
    from ...core import traffic as TR
    from ...core.project import Edit
    i, tp = C.find(ctx, 'traffic_profiles', profile, 'traffic profile')
    nt = copy.deepcopy(tp)
    for k, v in (('name', name), ('vehicles', vehicles), ('walkers', walkers), ('handedness', handedness)):
        if v is not None:
            nt[k] = v
    if fleet:
        nt['fleet'] = dict(nt.get('fleet') or {}, **{k: float(v) for k, v in fleet.items()})
    if behaviour:
        nt['behaviour'] = dict(nt.get('behaviour') or {}, **behaviour)
    if lateral_filtering is not None:
        nt['lateral_filtering'] = dict(TR.LATERAL_DEFAULTS, **(tp.get('lateral_filtering') or {}),
                                       **lateral_filtering)
        try:
            TR.lateral_settings(nt)
        except ValueError as e:
            raise ToolError(f'lateral filtering: {e}') from e
    return _validated(ctx, Edit('set', ['traffic_profiles', i], nt, text=f'Edit the traffic profile {tp["name"]}'))


@tool('project_traffic_set', 'project', 'act',
      'Change a traffic profile (Worlds ▸ Traffic): vehicle and walker counts, handedness (map, left, right), fleet '
      'shares by class (0–1), behaviour (distance_to_leading_m, ignore_lights_pct, …) and lateral filtering '
      '{enabled, vehicles [motorcycle, bicycle, autorickshaw], controller tm|own}. One undoable edit.',
      C.obj({'profile': C.s('the traffic profile id'), 'name': C.s('name'),
             'vehicles': C.i('background vehicles', minimum=0, maximum=500),
             'walkers': C.i('walkers', minimum=0, maximum=500),
             'handedness': C.s('which side traffic keeps to', enum=['map', 'left', 'right']),
             'fleet': {'type': 'object', 'description': 'class -> share 0–1',
                       'additionalProperties': {'type': 'number', 'minimum': 0, 'maximum': 1}},
             'behaviour': {'type': 'object', 'description': 'behaviour keys to set (project_options topic traffic)'},
             'lateral_filtering': {'type': 'object', 'description': 'two-wheelers between lanes: {enabled, '
                                                                    'vehicles, controller tm|own}'}}, ['profile']),
      preview=lambda ctx, **a: C.edit_preview(ctx, _traffic_set(ctx, **a)))
def project_traffic_set(ctx, **a):
    return _done(ctx, _traffic_set(ctx, **a))


# ================================================================================================= trigger rules
def _sensors(d) -> dict:
    return {s['id']: s['type'] for r in d.get('rigs') or [] for s in r.get('sensors') or []}


def _rules_check(d):
    from ...core import triggers as T
    return T.validate_rules(d.get('trigger_rules') or [], _sensors(d))


def _trigger_add(ctx, rule, id='', name='', repeat='once', cooldown_s=0.0, applies=None, enabled=True):
    from ...core import triggers as T
    from ...core.project import Edit
    d = C.doc(ctx)
    rules = d.get('trigger_rules') or []
    ids = {r['id'] for r in rules}
    if not id:
        k = len(rules) + 1
        while f'rule{k}' in ids:
            k += 1
        id = f'rule{k}'
    elif id in ids:
        raise ToolError(f'a rule {id!r} exists already')
    r = T.new_rule(rule.strip(), id, name, repeat, float(cooldown_s), tuple(applies or T.CONTEXTS), bool(enabled))
    errs = T.validate_rule(r, ids, _sensors(d))
    if errs:
        raise ToolError('refused by the rule check: ' + '; '.join(errs[:6]))
    if 'trigger_rules' not in d:
        e = Edit('set', ['trigger_rules'], [r], text=f'Add the trigger rule {id}')
    else:
        e = Edit('insert', ['trigger_rules'], r, text=f'Add the trigger rule {id}')
    return _validated(ctx, e, _rules_check)


_RULE = {'rule': C.s('"when <condition> -> <action>[; <action>]" (syntax: project_options topic triggers)'),
         'name': C.s('a short name'), 'repeat': C.s('once, every or cooldown', enum=['once', 'every', 'cooldown']),
         'cooldown_s': C.n('seconds between firings with repeat cooldown', minimum=0),
         'applies': C.arr(C.s('context', enum=['drives', 'scenarios', 'live']), 'where it applies (default all)'),
         'enabled': C.b('on (default true)')}


@tool('project_trigger_add', 'project', 'act',
      'Add a live trigger rule (Scenarios ▸ Trigger rules): "when ego.speed > 40 and visibility < 30 -> spawn '
      'jaywalker". Parsed and checked by core.triggers (the column of a syntax error is given); refused when '
      'invalid. Drives and scenario runs then carry the enabled rules that apply to them.',
      C.obj(dict(_RULE, id=C.s('its id (default ruleN)')), ['rule']),
      preview=lambda ctx, **a: C.edit_preview(ctx, _trigger_add(ctx, **a)))
def project_trigger_add(ctx, **a):
    e = _trigger_add(ctx, **a)
    return _done(ctx, e, rule=e.value[0]['id'] if isinstance(e.value, list) else e.value['id'])


def _trigger_set(ctx, id, **fields):
    from ...core import triggers as T
    from ...core.project import Edit
    i, r = C.find(ctx, 'trigger_rules', id, 'trigger rule')
    nr = copy.deepcopy(r)
    for k, v in fields.items():
        if v is not None:
            nr[k] = list(v) if k == 'applies' else (v.strip() if isinstance(v, str) else v)
    errs = T.validate_rule(nr, {x['id'] for x in C.doc(ctx)['trigger_rules']} - {id}, _sensors(C.doc(ctx)))
    if errs:
        raise ToolError('refused by the rule check: ' + '; '.join(errs[:6]))
    return _validated(ctx, Edit('set', ['trigger_rules', i], nr, text=f'Trigger rule {id}'), _rules_check)


@tool('project_trigger_set', 'project', 'act',
      'Change a trigger rule: its text, name, repeat, cooldown, contexts, or enable/disable it (enabled false). '
      'Checked by core.triggers; refused when invalid.', C.obj(dict(_RULE, id=C.s('the rule id')), ['id']),
      preview=lambda ctx, **a: C.edit_preview(ctx, _trigger_set(ctx, **a)))
def project_trigger_set(ctx, **a):
    return _done(ctx, _trigger_set(ctx, **a))


@tool('project_trigger_check', 'project', 'read',
      'Check trigger rules with core.triggers\' parser and checks: the project\'s rules (each problem names the '
      'rule), or one rule text before adding it (its parsed condition, actions and the signals it reads).',
      C.obj({'rule': C.s('a rule text to check instead of the project\'s rules')}))
def project_trigger_check(ctx, rule=''):
    from ...core import triggers as T
    if rule:
        try:
            p = T.parse_rule(rule)
        except T.RuleError as e:
            return {'ok': False, 'problems': [str(e)]}
        r = T.new_rule(rule, 'check')
        errs = T.validate_rule(r, (), _sensors(C.doc(ctx)))
        return {'ok': not errs, 'problems': errs, 'signals': sorted(p.get('signals') or []),
                'describes_as': C.clip(T.describe(r), 300)}
    rules = C.doc(ctx).get('trigger_rules') or []
    errs = _rules_check(C.doc(ctx))
    return {'ok': not errs, 'rules': len(rules), 'problems': errs[:30]}


# ================================================================================================= roadside units
@tool('project_rsu_junctions', 'project', 'read',
      'An imported world\'s junctions as places for a roadside unit (V2X): id, name (the OSM ways meeting there), '
      'centre (CARLA x, y), radius, approaches, whether signalised — signalised first; and the units already placed '
      'at each. Town worlds have none (place by position).',
      C.obj({'world': C.s('an imported world id'), 'limit': C.i('rows (default 15)', minimum=1, maximum=50)},
            ['world']))
def project_rsu_junctions(ctx, world, limit=15):
    from ...core import v2x as V
    _i, w = C.find(ctx, 'worlds', world, 'world')
    text, folder = _xodr(ctx, w)
    js = V.junctions(text, folder, (w.get('osm') or {}).get('centre'))
    used = {}
    for r in V.rsus(C.doc(ctx), w['id']):
        jid = (r.get('junction') or {}).get('id')
        if jid is not None:
            used[jid] = used.get(jid, 0) + 1
    rows = [{'id': j['id'], 'name': j.get('name'), 'x': round(j['x'], 1), 'y': round(j['y'], 1),
             'radius_m': round(j.get('radius_m', 0), 1), 'signalised': bool(j.get('signalised')),
             'approaches': len(j.get('approaches') or []), 'units_here': used.get(j['id'], 0)} for j in js]
    return C.capped(rows, int(limit), 'raise limit for more')


def _xodr(ctx, w):
    """(text, folder) of an imported world's OpenDRIVE file, as core.v2x.world_xodr reads it, checked first."""
    if w.get('kind') == 'town' or not w.get('xodr'):
        raise ToolError(f'{w.get("name")}: junction placement needs an imported (OpenDRIVE) world; place a unit by '
                        'position in a town')
    path = C.checked(ctx, os.path.join(C.workspace(ctx), w['xodr']))
    try:
        with open(path, encoding='utf-8') as fh:
            return fh.read(), os.path.dirname(path)
    except OSError as e:
        raise ToolError(f'cannot read the world\'s OpenDRIVE file: {e.strerror}') from e


def _rsu_place(ctx, world, junction=None, x=None, y=None, yaw=0.0, preset='camera-lidar', height_m=6.0, name=''):
    from ...core import v2x as V
    from ...core.project import Edit
    if preset not in V.PRESETS:
        raise ToolError(f'preset must be one of {", ".join(V.PRESETS)}')
    _i, w = C.find(ctx, 'worlds', world, 'world')
    if junction is not None:
        text, folder = _xodr(ctx, w)
        js = {str(j['id']): j for j in V.junctions(text, folder, (w.get('osm') or {}).get('centre'))}
        if str(junction) not in js:
            raise ToolError(f'no junction {junction!r} in {w["name"]} (project_rsu_junctions lists them)')
        used = sum(1 for r in V.rsus(C.doc(ctx), w['id'])
                   if str((r.get('junction') or {}).get('id')) == str(junction))
        try:
            r = V.place_at_junction(w['id'], js[str(junction)], text, preset, float(height_m), corner=used,
                                    name=name)
        except ValueError as e:
            raise ToolError(str(e)) from e
    elif x is not None and y is not None:
        r = V.new_rsu(w['id'], float(x), float(y), float(yaw), preset, float(height_m), name=name)
    else:
        raise ToolError('give a junction id, or x and y')
    d = C.doc(ctx)
    if V.KEY not in d:
        e = Edit('set', [V.KEY], {'rsus': [r]}, text=f'Add {r["name"]}')
    else:
        e = Edit('insert', [V.KEY, 'rsus'], r, text=f'Add {r["name"]}')
    return _validated(ctx, e, lambda dd: V.validate_all(dd) if hasattr(V, 'validate_all') else [])


@tool('project_rsu_place', 'project', 'act',
      'Place a roadside unit (smart pole: camera, lidar or both) in a world (V2X): at a junction of an imported world '
      '(at its next free corner, clear of the driving lanes, camera aimed past the centre) or at a CARLA position '
      'x, y, yaw. Drives in that world then record it and its Collective Perception Messages. Returns its id.',
      C.obj({'world': C.s('the world id'), 'junction': C.s('a junction id (project_rsu_junctions)'),
             'x': C.n('CARLA x (m)'), 'y': C.n('CARLA y (m)'), 'yaw': C.n('heading (deg)'),
             'preset': C.s('sensors', enum=['pole-camera', 'intersection-lidar', 'camera-lidar']),
             'height_m': C.n('mast height', minimum=1, maximum=30), 'name': C.s('its name')}, ['world']),
      preview=lambda ctx, **a: C.edit_preview(ctx, _rsu_place(ctx, **a)))
def project_rsu_place(ctx, **a):
    e = _rsu_place(ctx, **a)
    v = e.value['rsus'][0] if 'rsus' in (e.value or {}) else e.value
    return _done(ctx, e, rsu=v['id'])


def _rsu_set(ctx, rsu, name=None, x=None, y=None, yaw=None, height_m=None, cpm=None):
    from ...core import v2x as V
    from ...core.project import Edit
    rs = (C.doc(ctx).get(V.KEY) or {}).get('rsus') or []
    k = next((n for n, r in enumerate(rs) if r['id'] == rsu), None)
    if k is None:
        raise ToolError(f'no roadside unit {rsu!r}')
    nr = copy.deepcopy(rs[k])
    for f, v in (('name', name), ('x', x), ('y', y), ('yaw', yaw), ('height_m', height_m)):
        if v is not None:
            nr[f] = v if f == 'name' else round(float(v), 3)
    if cpm:
        c = dict(nr.get('cpm') or {})
        for f, v in cpm.items():
            c[f] = dict(c.get('noise') or {}, **v) if f == 'noise' else v
        nr['cpm'] = c
    return _validated(ctx, Edit('set', [V.KEY, 'rsus', k], nr, text=f'Edit {rs[k]["name"]}'),
                      lambda d: V.validate_rsu(d[V.KEY]['rsus'][k], [w['id'] for w in d['worlds']]))


@tool('project_rsu_set', 'project', 'act',
      'Change a roadside unit: name, position x/y, yaw, mast height, and its CPM settings (rate_hz, min_pixels, '
      'min_points, noise {position_m, velocity_mps}).',
      C.obj({'rsu': C.s('the unit id'), 'name': C.s('name'), 'x': C.n('x'), 'y': C.n('y'), 'yaw': C.n('yaw'),
             'height_m': C.n('height', minimum=1, maximum=30),
             'cpm': {'type': 'object', 'description': 'CPM settings to set'}}, ['rsu']),
      preview=lambda ctx, **a: C.edit_preview(ctx, _rsu_set(ctx, **a)))
def project_rsu_set(ctx, **a):
    return _done(ctx, _rsu_set(ctx, **a))


# ================================================================================================= ODDs
def _odd_loop(ctx, odd=None, example=False, model=None):
    from ...core import loop as L
    from ...core.project import Edit
    if model is not None:
        C.find(ctx, 'models', model, 'model')
        return Edit('set', ['loop', 'model'], model, text='Choose the loop\'s model'), []
    if example:
        odd, missing = L.example_odd(C.project(ctx))
    elif not isinstance(odd, dict):
        raise ToolError('give an ODD object, example true, or a model')
    _norm, problems = L.normalise_odd(odd, C.project(ctx))
    return _validated(ctx, Edit('set', ['loop', 'odd'], odd, text='Set the loop\'s ODD')), problems


def _odd_preview(ctx, target='loop', odd=None, example=False, model=None):
    if target == 'safety_case':
        return f'write odd.json in the workspace: {C._short(odd if odd else "the example ODD", 400)}'
    e, problems = _odd_loop(ctx, odd, example, model)
    return C.edit_preview(ctx, e) + (f'\nproblems: {"; ".join(problems[:5])}' if problems else '')


@tool('project_odd_set', 'project', 'act',
      'Set an operational design domain. target loop (the project\'s loop): the ODD object or example true (the '
      'Bengaluru monsoon-dusk ODD for this project), or model to choose the loop\'s model; problems are reported '
      'and the ODD kept, as the Loop page does. target safety_case: write the workspace\'s odd.json (the given ODD '
      'or the example), checked by core.safetycase.validate_odd (refused when invalid).',
      C.obj({'target': C.s('loop or safety_case', enum=['loop', 'safety_case']),
             'odd': {'type': 'object', 'description': 'the ODD'}, 'example': C.b('use the example ODD'),
             'model': C.s('loop only: a registered model id to choose instead')}, ['target']),
      preview=_odd_preview)
def project_odd_set(ctx, target, odd=None, example=False, model=None):
    if target == 'safety_case':
        from ...core import safetycase as SCS
        from ...core.provenance import write_json
        o = odd if odd else SCS.example_odd()
        errs = SCS.validate_odd(o)
        if errs:
            raise ToolError('refused by the ODD check: ' + '; '.join(errs[:6]))
        p = os.path.join(C.workspace(ctx), 'odd.json')
        if os.path.exists(p):
            C.checked(ctx, p)
        write_json(p, o)
        return {'written': 'odd.json', 'note': 'a file in the workspace (not undoable): the previous one was replaced'
                if odd else 'the example ODD'}
    e, problems = _odd_loop(ctx, odd, example, model)
    return _done(ctx, e, problems=problems[:10])


# ================================================================================================= removing
def _remove(ctx, kind, id, rig=None):
    from ...core.project import Edit
    d = C.doc(ctx)
    if kind == 'sensor':
        if not rig:
            raise ToolError('a sensor is removed from a rig: give rig')
        i, r, j, s = _find_sensor(ctx, rig, id)
        return Edit('remove', ['rigs', i, 'sensors'], index=j, text=f'{r["name"]}: remove {s["name"]}')
    if kind == 'rsu':
        rs = (d.get('infrastructure') or {}).get('rsus') or []
        k = next((n for n, r in enumerate(rs) if r['id'] == id), None)
        if k is None:
            raise ToolError(f'no roadside unit {id!r}')
        return Edit('remove', ['infrastructure', 'rsus'], index=k, text=f'Remove {rs[k]["name"]}')
    sec = SECTIONS[kind]
    i, it = C.find(ctx, sec, id, kind.replace('_', ' '))
    if kind in ('world', 'rig') and len(d[sec]) <= 1:
        raise ToolError(f'the project keeps at least one {kind}')
    if kind == 'traffic_profile':
        users = [w['name'] for w in d['worlds'] if w.get('traffic_profile') == it['id']]
        if users:
            raise ToolError(f'{it["name"]} is used by {", ".join(users)}: give them another profile first')
    return _validated(ctx, Edit('remove', [sec], index=i, text=f'Remove {it.get("name") or it["id"]}'))


@tool('project_remove', 'project', 'act',
      'Remove one item from the project (undoable in the window): a world, rig, project scenario, traffic profile '
      '(not while a world uses it), trigger rule, drive set, roadside unit, or a sensor of a rig. The project keeps '
      'at least one world and one rig. Never touches recorded runs (runs_trash moves those aside).',
      C.obj({'kind': C.s('what', enum=['world', 'rig', 'scenario', 'traffic_profile', 'trigger_rule', 'drive_set',
                                       'rsu', 'sensor']),
             'id': C.s('its id'), 'rig': C.s('for a sensor: its rig id')}, ['kind', 'id']),
      preview=lambda ctx, **a: C.edit_preview(ctx, _remove(ctx, **a)))
def project_remove(ctx, **a):
    return _done(ctx, _remove(ctx, **a))
