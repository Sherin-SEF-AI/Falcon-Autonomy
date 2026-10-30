"""World imports (standard library only): the configs the OSM import worker runs, and the world entries they add.

An imported world's files live in the workspace under worlds/<world id>/ (area.osm, raw.xodr, world.xodr, graph.json,
import.json, run.json); the project's world entry names world.xodr relative to the workspace and keeps the import's
source and settings in its "osm" block, so that the import can be re-run.
"""
from __future__ import annotations

import os

IMPORT_SCHEMA = 'falcon.osm_import/1'
DEFAULT_SETTINGS = {'lane_width': 3.35, 'traffic_lights': True, 'all_junction_lights': False, 'way_types': None,
                    'excluded_light_way_types': None, 'speed_breakers': 'osm', 'osm_bump': 'osm_bump'}
MESH_DEFAULTS = {'vertex_distance': 2.0, 'max_road_length': 500.0, 'wall_height': 0.0, 'additional_width': 0.6,
                 'smooth_junctions': True, 'enable_mesh_visibility': True, 'enable_pedestrian_navigation': True}

# the towns of CARLA 0.9.16's base package (the "_Opt" variants have layers that can be unloaded; Town10HD_Opt is the
# default map). Town06, 07 and 11–15 come with CARLA's additional-maps package: once a probe has listed the server's
# maps, those are offered instead.
TOWNS = ('Town01', 'Town01_Opt', 'Town02', 'Town02_Opt', 'Town03', 'Town03_Opt', 'Town04', 'Town04_Opt', 'Town05',
         'Town05_Opt', 'Town10HD', 'Town10HD_Opt')

def world_dir(workspace: str, world_id: str) -> str:
    return os.path.join(workspace, 'worlds', world_id)


def import_config(source: str, out_dir: str, bbox=None, lht: bool = True, settings: dict | None = None,
                  centre=None, seed: int = 0, lane_check_sample: int = 50, city: bool | dict = True) -> dict:
    """An OSM import's config. city: dress the world with stand-in buildings, trees, lamps and ground from the same
    source (core.cityprops; a dict gives its source, seed and options), False not to."""
    cfg = {'schema': IMPORT_SCHEMA, 'source': os.path.abspath(source), 'bbox': list(bbox) if bbox else None,
           'centre': list(centre) if centre else None, 'lht': bool(lht),
           'settings': dict(DEFAULT_SETTINGS, **(settings or {})), 'out': os.path.abspath(out_dir), 'seed': seed,
           'lane_check_sample': lane_check_sample}
    if city:
        cfg['city'] = dict({'seed': seed}, **(city if isinstance(city, dict) else {}))
    return cfg


def world_entry(world_id: str, name: str, cfg: dict, report: dict, mesh: dict | None = None,
                traffic_profile: str = 'tp-default') -> dict:
    return {'id': world_id, 'name': name, 'kind': 'osm', 'xodr': os.path.join('worlds', world_id, 'world.xodr'),
            'osm': {'source': cfg['source'], 'bbox': cfg['bbox'], 'centre': report.get('centre'), 'lht': cfg['lht'],
                    'settings': cfg['settings'],
                    'mesh': _mesh(dict(MESH_DEFAULTS, **(mesh or {})), report),
                    'handedness': report.get('handedness', 'RHT')},
            'weather': {'preset': 'ClearNoon'}, 'traffic_profile': traffic_profile,
            **({'city': dict(report['city'])} if (report or {}).get('city', {}).get('plan') else {})}


def _mesh(mesh: dict, report: dict) -> dict:
    """A world with road features is generated at the vertex distance that renders them (core.roadfeatures)."""
    from .roadfeatures import mesh_for
    return mesh_for(mesh, bool(((report or {}).get('road_features') or {}).get('placed')))


def imported_name(cfg: dict) -> str:
    """An imported world's name from its source file (Worlds ▸ Import place): 'bengaluru-centre.osm' -> 'Bengaluru
    Centre'."""
    return os.path.basename(cfg['source']).split('.')[0].replace('-', ' ').replace('_', ' ').title()


def imported_world(doc: dict, world_id: str, cfg: dict, report: dict, mesh: dict | None = None,
                   name: str = '') -> dict:
    """The world entry an OSM import adds to the project (Worlds ▸ Import place ▸ Add world): its traffic profile
    the project's first left-hand one when the import is left-hand traffic, else CARLA's defaults."""
    left = next((tp['id'] for tp in doc['traffic_profiles'] if tp.get('handedness') == 'left'), None)
    tp = left if report.get('handedness') == 'LHT' and left else 'tp-default'
    return world_entry(world_id, name or imported_name(cfg), cfg, report, mesh, tp)


def edited_world(world: dict, cfg: dict, report: dict, workspace: str) -> dict:
    """The world after its map edits were applied by a re-conversion into cfg['out'] (the map editor's Apply): the
    new OpenDRIVE file, the edit list and its hash, the measured handedness and the mesh road features need."""
    import copy
    from . import mapedit as ME
    from .roadfeatures import mesh_for
    nw = copy.deepcopy(world)
    nw['xodr'] = os.path.relpath(os.path.join(cfg['out'], 'world.xodr'), workspace)
    nw['osm']['edits'] = list(cfg['edits'])
    nw['osm']['applied_edits'] = ME.edits_hash(cfg['edits'])
    nw['osm']['handedness'] = report.get('handedness', nw['osm'].get('handedness'))
    nw['osm']['mesh'] = mesh_for(nw['osm'].get('mesh'), bool((report.get('road_features') or {}).get('placed')))
    return nw
